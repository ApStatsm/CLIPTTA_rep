from typing import Dict, Any, Optional, Tuple, List
from typing_extensions import TypeAlias

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from ttavlm.methods.cliptta_otsu import CLIPTTA
from ttavlm.models.clip import tokenize as clip_tokenize

import ttavlm.lib as lib

Kwargs: TypeAlias = Dict[str, Any]

_EPS = 1e-12


class CLIPTTA_Prior(CLIPTTA):
    """
    CLIPTTA with a dynamic class-prior.

    This method keeps the exact CLIPTTA (soft contrastive) loss structure but
    replaces the pseudo-label source with a Bayesian posterior:

        likelihood = softmax(image_text_logits)
        posterior  = normalize(likelihood * prior)
        pseudo_label = posterior.argmax(dim=-1)

    The class-prior ``pi`` is initialized as a uniform distribution and is
    updated after each adaptation step using only the samples whose confidence
    actually improved (``gain``-filtered, ``gain``-weighted average).

    The baseline CLIPTTA implementation (``cliptta_otsu.CLIPTTA``) is left
    untouched; this class only overrides the pseudo-label computation and adds
    the prior bookkeeping/update.
    """

    def __init__(
        self,
        template: List[str],
        class_names: List[str],
        prior_alpha: float = 0.9,
        prior_strength: float = 1.0,
        prior_min_gain: float = 0.0,
        prior_min_conf: float = 0.0,
        use_dynamic_prior: bool = False,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(template=template, class_names=class_names, **kwargs)

        self.prior_alpha = prior_alpha
        self.prior_strength = prior_strength
        self.prior_min_gain = prior_min_gain
        self.prior_min_conf = prior_min_conf
        self.use_dynamic_prior = use_dynamic_prior

        self.num_classes = len(class_names)
        device = self.class_prototypes.device

        # Class prior pi, initialized as a uniform distribution.
        self.register_buffer("prior", torch.ones(self.num_classes, device=device) / self.num_classes)

        # Bookkeeping populated during a forward pass.
        self.posterior_before: Optional[Tensor] = None
        self.conf_before: Optional[Tensor] = None
        self.posterior_after: Optional[Tensor] = None
        self.conf_after: Optional[Tensor] = None

    # ------------------------------------------------------------------ #
    # Prior helpers
    # ------------------------------------------------------------------ #
    def _sanitize(self, p: Tensor) -> Tensor:
        """Replace NaN/Inf, clamp to non-negative and renormalize to sum=1.

        Falls back to a uniform distribution if the input is degenerate.
        """
        p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        p = p.clamp(min=0.0)
        s = p.sum()
        if (not torch.isfinite(s)) or (s <= _EPS):
            return torch.ones_like(p) / p.numel()
        return p / s

    def _current_prototypes(self) -> Tensor:
        if self.update_text:
            class_prototypes, _ = lib.get_text_features(self.class_names, self.template, self.clip_text_encoder)
            return class_prototypes
        return self.class_prototypes

    def _compute_posterior(self, image_features: Tensor, class_prototypes: Tensor) -> Tuple[Tensor, Tensor]:
        """Return (likelihood, posterior) for a batch of image features."""
        logits = self.logit_scale * (image_features @ class_prototypes.t())
        likelihood = logits.softmax(dim=-1)
        likelihood = torch.nan_to_num(likelihood, nan=0.0, posinf=0.0, neginf=0.0)

        prior = self._sanitize(self.prior.to(image_features.device))
        # prior_strength controls how strongly the prior biases the posterior.
        prior_weight = prior.clamp(min=_EPS) ** self.prior_strength

        unnorm = likelihood * prior_weight.unsqueeze(0)
        denom = unnorm.sum(dim=-1, keepdim=True)
        posterior = unnorm / (denom + _EPS)
        posterior = torch.nan_to_num(posterior, nan=0.0, posinf=0.0, neginf=0.0)

        return likelihood, posterior

    def _record_prior_snapshot(self, selected_ratio: float, num_selected: int, prior_delta_l1: float) -> None:
        """Record current prior statistics for CSV logging."""
        p = self.prior.detach()
        n = p.numel()
        pc = p.clamp(min=_EPS)
        prior_entropy = float(-(pc * pc.log()).sum().item())
        prior_entropy_norm = prior_entropy / math.log(n) if n > 1 else 0.0
        self.record_prior_stats(
            prior_entropy=prior_entropy,
            prior_entropy_norm=prior_entropy_norm,
            top1_prior_class=int(p.argmax().item()),
            top1_prior_value=float(p.max().item()),
            prior_delta_l1=float(prior_delta_l1),
            selected_ratio=float(selected_ratio),
            num_selected=int(num_selected),
        )

    @torch.no_grad()
    def _update_prior(self, images: List[Tensor]) -> None:
        """Update the class-prior from the post-adaptation posterior."""
        class_prototypes = self._current_prototypes()
        image_features = self.get_features(images)
        _, posterior_after = self._compute_posterior(image_features[0], class_prototypes)
        conf_after = posterior_after.max(dim=-1).values

        self.posterior_after = posterior_after.detach()
        self.conf_after = conf_after.detach()

        num_selected = 0
        selected_ratio = 0.0
        prior_delta_l1 = 0.0

        conf_before = self.conf_before
        if conf_before is not None and conf_before.shape[0] == conf_after.shape[0]:
            # gain = conf_after - conf_before
            gain = conf_after - conf_before.to(conf_after.device)

            # Keep only samples that improved and are confident enough.
            mask = (gain > self.prior_min_gain) & (conf_after > self.prior_min_conf)
            num_selected = int(mask.sum().item())
            selected_ratio = num_selected / max(conf_after.shape[0], 1)

            if num_selected > 0:
                selected_posterior = posterior_after[mask]
                weights = gain[mask].clamp(min=0.0)
                wsum = weights.sum()
                if (not torch.isfinite(wsum)) or (wsum <= _EPS):
                    # Degenerate weights -> fall back to a plain average.
                    weights = torch.ones_like(weights)
                    wsum = weights.sum()

                # gain-weighted average of the selected posteriors.
                batch_prior = (weights.unsqueeze(1) * selected_posterior).sum(dim=0) / wsum
                batch_prior = self._sanitize(batch_prior)

                old_prior = self.prior.clone()
                new_prior = self.prior_alpha * self.prior + (1.0 - self.prior_alpha) * batch_prior
                new_prior = self._sanitize(new_prior)
                prior_delta_l1 = float((new_prior - old_prior).abs().sum().item())
                self.prior = new_prior

        self._record_prior_snapshot(selected_ratio, num_selected, prior_delta_l1)

    # ------------------------------------------------------------------ #
    # CLIPTTA hooks
    # ------------------------------------------------------------------ #
    def before_adaptation(self, images: List[Tensor], **kwargs: Kwargs) -> None:
        # Preserve baseline behaviour (e.g. Otsu init).
        super().before_adaptation(images, **kwargs)

        with torch.no_grad():
            class_prototypes = self._current_prototypes()
            image_features = self.get_features(images)
            _, posterior_before = self._compute_posterior(image_features[0], class_prototypes)
            self.posterior_before = posterior_before.detach()
            self.conf_before = posterior_before.max(dim=-1).values.detach()

    def compute_loss_tta(self, image_features: List[Tensor], class_prototypes: Tensor) -> Tensor:
        feats = image_features[0]
        logits = feats @ class_prototypes.t()

        # Posterior-based pseudo-labels (the only change vs. baseline CLIPTTA).
        _, posterior = self._compute_posterior(feats, class_prototypes)
        pred = posterior.argmax(dim=-1)
        pred_text_features = class_prototypes[pred]

        # Compute logits (image v.s. pseudo-captions, i.e. size is B x B)
        logits_per_image = self.logit_scale * feats @ pred_text_features.t()
        logits_per_text = logits_per_image.t() if self.update_text else logits_per_image

        # TTA loss (identical structure to baseline CLIPTTA).
        if self.use_tent:
            loss_tta = lib.softmax_entropy(self.logit_scale * logits).mean(0)
        elif self.use_clipartt:
            _, topk_pred = logits.topk(self.K, 1, True, True)
            if self.K == 1:
                text_features = self.class_prototypes[topk_pred[:, 0]]
            else:
                text_prompts = lib.getprompt(self.K, topk_pred.cpu().numpy(), self.class_names, self.template[0])
                pred_inputs = clip_tokenize(text_prompts).to(logits.device)

                with torch.no_grad():
                    text_features = self.clip_text_encoder(pred_inputs)
                    text_features = text_features / text_features.norm(dim=1, keepdim=True)

            images_similarity = feats @ feats.t()
            texts_similarity = text_features @ text_features.t()
            targets = F.softmax(((images_similarity + texts_similarity) / 2) / self.clipartt_temp, dim=-1)

            predictions = (self.logit_scale * text_features @ feats.t()).t()
            loss_tta = F.cross_entropy(predictions, targets)
        else:
            if self.use_softmax_entropy:
                loss_tta = (lib.softmax_entropy(logits_per_image).mean(0) + lib.softmax_entropy(logits_per_text).mean(0)) / 2
            else:
                targets = torch.eye(logits_per_image.shape[0]).to(logits_per_image.device)
                loss_tta = (self.loss_fn(logits_per_image, targets).mean(0) + self.loss_fn(logits_per_text, targets).mean(0)) / 2

        return loss_tta

    @torch.enable_grad()
    def forward_and_adapt(
        self,
        images: List[Tensor],
        step: int,
        labels: Tensor = None,
    ) -> Tuple[List[Tensor], Tensor]:
        logits, scores = super().forward_and_adapt(images, step, labels)

        # After adaptation: compute posterior_after / conf_after and update prior.
        if step == self.steps - 1:
            if self.use_dynamic_prior:
                self._update_prior(images)
            else:
                # Prior is unchanged, but still log its current statistics so the
                # prior columns are populated (not NA) for this prior-based method.
                self._record_prior_snapshot(selected_ratio=0.0, num_selected=0, prior_delta_l1=0.0)

        return logits, scores

    def _reset_extra(self) -> None:
        super()._reset_extra()
        # Reset the prior to a uniform distribution between episodes/runs.
        self.prior = torch.ones(self.num_classes, device=self.prior.device) / self.num_classes
        self.posterior_before = None
        self.conf_before = None
        self.posterior_after = None
        self.conf_after = None
