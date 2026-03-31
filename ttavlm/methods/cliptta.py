from typing import Dict, Any, Tuple, List
from typing_extensions import TypeAlias

from functools import partial

import torch
import torch.nn as nn
from torch import Tensor
from copy import deepcopy

from ttavlm.methods.abstract_model import AbstractOpenSetTTAModel
from ttavlm.memory import CCM

import ttavlm.lib as lib

Kwargs: TypeAlias = Dict[str, Any]


class CLIPTTA_Old(AbstractOpenSetTTAModel):
    """
    CLIPTTA adapts CLIP using the same loss as during the pre-training.
    """

    def __init__(
        self,
        template: List[str],
        class_names: List[str],
        use_softmax_entropy: bool = False,
        use_memory: bool = False,
        use_scheduler: bool = False,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.class_names = class_names
        self.template = template
        self.use_softmax_entropy = use_softmax_entropy
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.use_memory = use_memory
        self.use_scheduler = use_scheduler

        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_iter)

        if self.use_memory:
            self.memory = CCM(num_shots=self.num_shots, num_classes=len(class_names), sample_size=max(self.sample_size, len(class_names)))

        if self.measure_improvement:
            self.model0 = deepcopy(self.model)
            for param in self.model0.parameters():
                param.detach()

    @torch.enable_grad()
    def _forward_and_adapt(
        self,
        images: List[Tensor],
    ) -> Tensor:
        if self.update_text:    # 텍스트 업데이트?
            class_prototypes, _ = lib.get_text_features(self.class_names, self.template, self.clip_text_encoder, enable_grad=True)  # 텍스트 특징 얻기 (업데이트 가능)
        else:
            class_prototypes = self.class_prototypes # 텍스트 업데이트 안함

        image_features = self.get_features(images) # 이미지 특징 얻기 (업데이트 가능)
        similarity = image_features[0] @ class_prototypes.t() # 이미지-텍스트 유사도 계산

        _, pred = similarity.topk(1, 1, True, True) # topk 클래스 예측 (가장 유사한 텍스트 특징의 인덱스)
        pred_text_features = class_prototypes[pred[:, 0]]   # 각 이미지별로 예측한 클래스 가져와서 텍스트 임베딩들 나열.

        # Obtain new logits (image v.s. new prompt, i.e. size is B x B)
        logits_per_image = self.logit_scale * image_features[0] @ pred_text_features.t()    # 이미지 하나에 대한 텍스트들과의 유사도
        logits_per_text = logits_per_image.t() if self.update_text else logits_per_image    # 텍스트 업데이트 안하면 이미지-텍스트 유사도 그대로 사용, 업데이트 하면 텍스트-이미지 유사도로 전치해서 사용

        if self.use_softmax_entropy: # 소프트맥스 엔트로피 쓰면
            loss = ((lib.softmax_entropy(logits_per_image)).mean(0) + (lib.softmax_entropy(logits_per_text)).mean(0)) / 2   # 이미지-텍스트랑 텍스트-이미지 둘다 소프트맥스 엔트로피로 loss 만들어서 계산.
        else:   
            targets = torch.eye(images[0].shape[0]).to(images[0].device)    # target = I 행렬 생성 (constrastive)
            loss = ((self.loss_fn(logits_per_image, targets)).mean(0) + (self.loss_fn(logits_per_text, targets)).mean(0)) / 2   
            # 이미지-텍스트랑 텍스트-이미지 둘다 I 행렬이랑 loss 만들어서 계산. (contrastive loss)

        loss -= self.beta_reg * lib.softmax_mean_entropy(self.logit_scale * similarity) # 배치 평균 softmax 엔트로피로 키우는 방향으로 정규화
        loss.backward() # 역전파
        return loss # loss 반환 (backward()로 계산된 그래디언트는 optimizer.step()에서 사용됨)

    @torch.enable_grad()
    def forward_and_adapt(
        self,
        images: List[Tensor],
        step: int,
        labels: Tensor = None,
    ) -> Tuple[List[Tensor], Tensor]:
        if step == 0 and self.use_memory:   # 첫스텝, memory 쓰면 => 현재 배치 정보를 memory에 업데이트
            with torch.no_grad():   # memory 업데이트는 그래디언트 계산 필요 없으니까 no_grad
                if self.update_text:    # 텍스트 업데이트 하면
                    text_features, _ = lib.get_text_features(self.class_names, self.template, self.clip_text_encoder, enable_grad=False)
                    # 텍스트 특징 얻어서 memory 업데이트할 때 사용 (업데이트 안하면 기존 텍스트 특징 그대로 사용)
                else:
                    text_features = self.class_prototypes  # 텍스트 업데이트 안하면 기존 텍스트 특징 그대로 사용
                image_features = self.get_features(images)  # 현재 이미지 배치에서 feature 추출
                logits = [image_feature @ text_features.t() for image_feature in image_features]    # 이미지-텍스트 유사도 행렬
                pred = logits[0].topk(1, 1, True, True)[1][:, 0]    # 그중에서 가장 큰 유사도
                scores = self.get_scores(logits, image_features)    # ??? 유사도 기반 OOD 점수 계산

            self.memory.update(images[0].cpu().detach(), pred.cpu().detach(), scores.cpu().detach())    # gradient 계산 안하니까 숫자만 저장.
        # import ipdb; ipdb.set_trace()
        adapt_samples = [self.memory.sample()[0].cuda(non_blocking=True)] if self.use_memory else images    # memory 쓰면 memory에서 샘플링한 이미지로 적응, 안쓰면 현재 배치 이미지로 적응
        _ = self._forward_and_adapt(adapt_samples)  # 그리고 adaptation step 수행 (그래디언트 계산됨)

        closure = partial(self._forward_and_adapt, images=adapt_samples) if self.use_sam else None  # SAM 쓰면 closure 만들어서 optimizer.step()에 넘겨줌, 안쓰면 None
        # 다른 optimizer는 필요없음

        self.optimizer.step(closure)    # 실제 업데이트 수행.
        self.optimizer.zero_grad(set_to_none=True)  # gardiant 초기화

        # Get final logits and OOD scores
        if step == self.steps - 1:  # 마지막 스텝에서
            with torch.no_grad():   # 평가만 할거임.
                if self.update_text:    # 위랑 똑같이 텍스트 임베딩 준비.
                    text_features, _ = lib.get_text_features(self.class_names, self.template, self.clip_text_encoder, enable_grad=False)
                else:
                    text_features = self.class_prototypes
                image_features = self.get_features(images)  # 이미지도 준비
                logits = [image_feature @ text_features.t() for image_feature in image_features]    # 이미지-텍스트 유사도 계산
                scores = self.get_scores(logits, image_features)    # 점수 계산.
        else:
            logits, scores = None, None # 마지막 스텝이 아니면 평가 안하니까 None 반환.
        return logits, scores   # logits는 이미지-텍스트 유사도, scores는 OOD 점수임. 

    def compute_loss(self, image_features: Tensor) -> Tensor:
        similarity = image_features @ self.class_prototypes.t()
        _, pred = similarity.topk(1, 1, True, True)
        pred_text_features = self.class_prototypes[pred[:, 0]]  # 모델이 각 샘플을 어떤 클래스로 보는지.(pseudo-label)
        logits_per_image = self.logit_scale * image_features @ pred_text_features.t()   # 유사도행렬
        loss = (lib.softmax_entropy(logits_per_image)).mean(0)  # 각 샘플 row마다 소프트맥스 엔트로피 계산하고 평균
        loss -= self.beta_reg * lib.softmax_mean_entropy(self.logit_scale * similarity) # 배치 평균 softmax 엔트로피를 키우는 방향으로 정규화
        return loss # loss 반환 

    def after_adaptation(self, **kwargs: Kwargs) -> None:
        if self.use_scheduler:  # scheduler 쓰면 스텝마다 scheduler 업데이트
            self.scheduler.step()
