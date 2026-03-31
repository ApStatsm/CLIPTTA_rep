from typing import List
from argparse import Namespace as ArgsType

from torch import nn

from ttavlm.methods.abstract_model import AbstractOpenSetTTAModel
from ttavlm.methods.clipartt import CLIPArTT
from ttavlm.methods.cliptta_otsu import CLIPTTA
from ttavlm.methods.cliptta import CLIPTTA_Old
from ttavlm.methods.source import SourceModel
from ttavlm.methods.stamp import STAMP
from ttavlm.methods.tent import Tent, TentOracle
from ttavlm.methods.lame import Lame
from ttavlm.methods.eata import ETA
from ttavlm.methods.sar import SAR
from ttavlm.methods.calip import CALIP
from ttavlm.methods.tda import TDA
from ttavlm.methods.unient import UniEnt
from ttavlm.methods.ostta import OSTTA
from ttavlm.methods.rotta import RoTTA
from ttavlm.methods.sotta import SoTTA
from ttavlm.methods.adacontrast import AdaContrast
from ttavlm.methods.watt import Watt, WattOtsu, WattUniEnt
from ttavlm.methods.zero import Zero

from ttavlm.lib import negative_classes
from ttavlm.lib.prompts import get_text_features
from ttavlm.models.clip import CLIPTextEncoder
from ttavlm.lib import LOGGER

__all__ = [
    "AbstractOpenSetTTAModel",
    "CLIPArTT",
    "CLIPTTA",
    "CLIPTTA_Old",
    "OSTTA",
    "SourceModel",
    "STAMP",
    "Tent",
    "TentOracle",
    "Lame",
    "ETA",
    "SAR",
    "CALIP",
    "TDA",
    "UniEnt",
    "RoTTA",
    "SoTTA",
    "AdaContrast",
    "Watt",
    "WattOtsu",
    "WattUniEnt",
    "Zero",
    "return_tta_model",
]   # 불러올것들 목록?


def return_tta_model(
    model_type: str,
    base_model: nn.Module,
    args: ArgsType,
    template: List[str] = ["a photo of a {}"],
    class_names: List[str] = None,
) -> AbstractOpenSetTTAModel:
    LOGGER.info(f"Loading {model_type}")
    LOGGER.info(f"Using template {template[0]}")
    if args.base_model_name.startswith("clip"): # CLIP 모델이면 텍스트 특징 얻어서 TTA 모델에 넘겨주기
        class_prototypes, class_bias = get_text_features(class_names, template, base_model)

        if args.score_type == "neglabel":   # OOD 점수 계산할 때 negative 클래스 사용하면 => negative 클래스 텍스트 특징도 얻어서 TTA 모델에 넘겨주기
            negative_prototypes, negative_bias = get_text_features(negative_classes[args.dataset], template, base_model)
        else:
            negative_prototypes = None
            negative_bias = None

        clip_text_encoder = CLIPTextEncoder(base_model) # Text encoder는 CLIP 모델에서 텍스트 인코더 부분만 따로 빼서 사용
        base_model = base_model.visual  # Vision encoder 따로 빼서 사용
        base_model.dtype = clip_text_encoder.dtype  # CLIP 모델의 텍스트 인코더와 비전 인코더가 같은 dtype을 사용하도록 맞춰주기
        normalize_features = True   # CLIP 모델은 특징 벡터가 L2 정규화되어 있으니까 TTA 모델에서도 특징 벡터 정규화해서 사용하기
        base_model.use_local = model_type == "calip"  # If using CALIP, return local and global features (otherwise only global)
    else:   # CLIP 모델 아니면 그냥 classifier로 사용.
        clip_text_encoder = None
        class_prototypes = base_model.fc.weight
        class_bias = base_model.fc.bias
        negative_prototypes = None
        negative_bias = None
        normalize_features = False

    # TTA 모델에 넘겨줄 공통 인자들
    base_tta_kwargs = {
        "save_root": args.save_root,    # 결과 저장 경로
        "adaptation": args.adaptation,  # adaptation 방법 이름
        "model": base_model.eval() if model_type == "source" else base_model,   # source 모델이면 TTA 안하고 바로 평가할거니까 모델을 eval 모드로 만들어서 넘겨주기. 나머지 모델은 TTA 하면서 평가할거니까 모델 그대로 넘겨주기.
        "clip_text_encoder": clip_text_encoder, # 클립 텍스트 인코더
        "class_prototypes": class_prototypes,   # 클래스별 텍스트 특징 벡터 (CLIP 모델이면 프롬프트로 얻은 텍스트 특징, 아니면 그냥 분류기 가중치)
        "class_bias": class_bias,   # 클래스별 바이어스 (CLIP 모델이면 0, 아니면 분류기 바이어스)
        "normalize_features": normalize_features,   # 특징 벡터 정규화 여부 (CLIP 모델이면 True, 아니면 False)
        "update_text": args.update_text,    # 이미지 특징 뿐만 아니라 텍스트 특징도 업데이트할지 여부
        "update_all_params": args.update_all_params,    # CLIP 모델이면 텍스트 인코더도 업데이트할 수 있으니까 모든 파라미터 업데이트할지 여부
        "optimizer_type": args.optimizer_type,  # optimizer 종류
        "steps": 1 if model_type in ["source", "lame", "calip", "tda", "watt", "watt_otsu", "watt_unient", "zero"] else args.steps,
        # source 모델은 TTA 안하니까 스텝 1로 고정(이 모델들은 스텝마다 업데이트하는 방식이 아니라 배치마다 업데이트하는 방식), 나머지 모델들은 args.steps로 설정
        "episodic": args.episodic,  # 에피소드 단위로 TTA 할지 여부 (True면 매 에피소드마다 모델 초기화, False면 계속 업데이트)
        "logit_scale": args.logit_scale,    # open-set
        "id_score_type": args.id_score_type,
        "ood_logit_scale": args.ood_logit_scale,
        "use_ood_loss": args.use_ood_loss,
        "detect_ood": args.detect_ood,
        "score_type": args.score_type,
        "loss_ood_type": args.loss_ood_type,
        "use_weights": args.use_weights,    # 용도
        "update_alpha": args.update_alpha,  # 기본 alpha 업데이트 여부
        "update_alpha_miss": args.update_alpha_miss,    # miss alpha 업데이트 여부
        "alpha": args.alpha,    # alpha 값
        "beta_tta": args.beta_tta, # TTA loss 가중치
        "beta_reg": args.beta_reg, # 정규화 항 가중치 (CLIP 모델은 배치 평균 softmax 엔트로피 키우는 방향으로 정규화, 다른 모델은 L2 정규화)
        "beta_ood": args.beta_ood, # OOD loss 가중치
        "gamma": args.gamma, # CLIPTTA에서 OOD 샘플에 대한 가중치
        "beta_schedule": args.beta_schedule, # beta_tta, beta_reg, beta_ood 스텝마다 어떻게 변화시킬지 (예: "linear"면 선형적으로 증가)
        "milestone": args.milestone, # beta_schedule이 "step"일 때 beta_tta, beta_reg, beta_ood를 증가시킬 스텝 번호 리스트
        "lr": args.lr, # 학습률
        "lr_miss": args.lr_miss, # miss 샘플에 대한 학습률 (CLIPTTA에서 OOD 샘플과 구분해서 업데이트할 때 사용)
        "momentum": args.momentum, # 모멘텀
        "weight_decay": args.weight_decay, # 가중치 감쇠
        "use_sam": args.use_sam, # SAM optimizer 사용할지 여부
        "skip_top_layers": args.skip_top_layers, # 몇 개의 top 레이어를 TTA에서 업데이트하지 않고 고정할지
        "max_iter": args.max_iter,  # TTA 최적화할 때 최대 몇 번 반복할지 (early stopping 용도)
        "use_batch_stats_only": args.use_batch_stats_only,  # batch stats만 사용할지
        "negative_prototypes": negative_prototypes, # negative 클래스 텍스트 특징 벡터 (CLIP 모델이면 프롬프트로 얻은 텍스트 특징, 아니면 None)
        "negative_bias": negative_bias,   # negative 클래스 바이어스 (CLIP 모델이면 0, 아니면 None)
        "distributed": args.distributed,    # 분산 학습 여부 (DistributedDataParallel 사용할 때는 True)
        "sample_size": args.sample_size,    # TTA 할 때 샘플링할 이미지 개수 (메모리 문제로 전체 배치가 아니라 일부 샘플로 TTA 하는 경우)
        "num_shots": args.num_shots,    # 몇 샷 프롬프트 사용할지 (CLIP 모델에서 프롬프트로 텍스트 특징 얻을 때 사용할 샷 수)
        "tsne": args.tsne,  # TTA 과정에서 특징 벡터 시각화할지 여부 (True면 TTA 과정에서 특징 벡터를 t-SNE로 시각화해서 저장)
        "measure_collapse": args.measure_collapse, # 특징 벡터 collapse 현상 측정할지 여부 (True면 TTA 과정에서 특징 벡터의 collapse 정도 측정해서 저장)
        "measure_improvement": args.measure_improvement,    # 성능 개선 측정할지 여부 (True면 TTA 과정에서 모델 업데이트하기 전과 후의 성능 비교해서 저장)
    }
    if model_type == "source":  # 소스모델이면 적응없이 평가
        model = SourceModel(**base_tta_kwargs)
    elif model_type == "tent":  # tent 모델이면 텐트 모델로 초기화
        model = Tent(**base_tta_kwargs)
    elif model_type == "tent_oracle": # tent oracle 모델이면 텐트 오라클 모델로 초기화
        model = TentOracle(
            oracle_miss=args.oracle_miss,
            oracle_ood=args.oracle_ood,
            miss_weight=args.miss_weight,
            **base_tta_kwargs,
        )
    elif model_type == "unient":
        model = UniEnt(
            use_cliptta_loss=args.use_cliptta_loss,
            use_clipartt_loss=args.use_clipartt_loss,
            template=template,
            class_names=class_names,
            K=args.K,
            clipartt_temp=args.clipartt_temp,
            use_memory=args.use_memory,
            **base_tta_kwargs,
        )
    elif model_type == "lame":
        model = Lame(
            affinity=args.affinity,
            **base_tta_kwargs,
        )
    elif model_type == "eta":
        model = ETA(
            d_margin=args.d_margin,
            alpha_entropy=args.alpha_entropy,
            **base_tta_kwargs,
        )
    elif model_type == "sar":
        model = SAR(
            reset_constant_em=args.reset_constant_em,
            alpha_entropy=args.alpha_entropy,
            **base_tta_kwargs,
        )
    elif model_type == "ostta":
        model = OSTTA(
            margin=args.margin_ostta,
            **base_tta_kwargs,
        )
    elif model_type == "clipartt":
        model = CLIPArTT(
            class_names=class_names,
            template=template,
            temp=args.clipartt_temp,
            K=args.K,
            **base_tta_kwargs,
        )
    elif model_type == "cliptta":
        model = CLIPTTA(
            template=template,
            class_names=class_names,
            use_softmax_entropy=args.use_softmax_entropy,
            use_memory=args.use_memory,
            use_scheduler=args.use_scheduler,
            use_tent=args.use_tent,
            use_clipartt=args.use_clipartt_loss,
            K=args.K,
            clipartt_temp=args.clipartt_temp,
            **base_tta_kwargs,
        )
    elif model_type == "cliptta_old":
        model = CLIPTTA_Old(
            template=template,
            class_names=class_names,
            use_softmax_entropy=args.use_softmax_entropy,
            use_memory=args.use_memory,
            use_scheduler=args.use_scheduler,
            **base_tta_kwargs,
        )
    elif model_type == "stamp":
        model = STAMP(
            memory_length=args.memory_length,
            alpha_stamp=args.alpha_stamp,
            use_consistency_filtering=args.use_consistency_filtering,
            **base_tta_kwargs,
        )
    elif model_type == "calip":
        model = CALIP(
            beta_calip=args.beta_calip,
            **base_tta_kwargs,
        )
    elif model_type == "tda":
        model = TDA(
            pos_alpha_beta=args.pos_alpha_beta,
            neg_alpha_beta=args.neg_alpha_beta,
            pos_shot_capacity=args.pos_shot_capacity,
            neg_shot_capacity=args.neg_shot_capacity,
            entropy_threshold=args.entropy_threshold,
            mask_threshold=args.mask_threshold,
            **base_tta_kwargs,
        )
    elif model_type == "rotta":
        model = RoTTA(
            capacity=args.capacity,
            update_frequency=args.update_frequency,
            lambda_u=args.lambda_u,
            lambda_t=args.lambda_t,
            alpha_rotta=args.alpha_rotta,
            nu=args.nu,
            use_tta=args.use_tta,
            **base_tta_kwargs,
        )
    elif model_type == "sotta":
        model = SoTTA(
            capacity=args.capacity,
            high_threshold=args.high_threshold,
            **base_tta_kwargs,
        )
    elif model_type == "adacontrast":
        model = AdaContrast(
            beta_ins=args.beta_ins,
            aug_type=args.aug_type,
            queue_size=args.queue_size,
            n_neighbors=args.n_neighbors,
            m=args.m,
            T_moco=args.T_moco,
            **base_tta_kwargs,
        )
    elif model_type == "watt":
        model = Watt(
            class_names=class_names,
            template=template,
            avg_type=args.avg_type,
            reps=args.reps,
            meta_reps=args.meta_reps,
            **base_tta_kwargs,
        )
    elif model_type == "watt_otsu":
        model = WattOtsu(
            class_names=class_names,
            template=template,
            avg_type=args.avg_type,
            reps=args.reps,
            meta_reps=args.meta_reps,
            **base_tta_kwargs,
        )
    elif model_type == "watt_unient":
        model = WattUniEnt(
            class_names=class_names,
            template=template,
            avg_type=args.avg_type,
            reps=args.reps,
            meta_reps=args.meta_reps,
            **base_tta_kwargs,
        )
    elif model_type == "zero":
        model = Zero(
            zero_gamma=args.zero_gamma,
            **base_tta_kwargs,
        )

    else:
        raise NotImplementedError

    return model
    # 모델들 초기화하고 반환하는 함수. 모델마다 필요한 인자들이 다르지만 공통적으로 base_tta_kwargs에 있는 인자들은 거의 다 필요해서 base_tta_kwargs로 묶어서 넘겨주고, 모델마다 추가로 필요한 인자들은 각 모델 초기화할 때 따로 넘겨주는 방식.
