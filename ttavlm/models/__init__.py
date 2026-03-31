from typing import Optional, Tuple, Callable
from os.path import join

import timm

import torch
import torch.nn as nn
from torch import device
from torchvision import transforms

from ttavlm.models.clip import load as load_clip
from ttavlm.lib import LOGGER

from ttavlm.models.clip import available_models, load, tokenize, CLIP

# 바깥으로 공개할 이름들..
__all__ = [
    "available_models",
    "load",
    "tokenize",
    "CLIP",
    "return_base_model",
]

# cifar용 Resnet50 모델 가중치 경로와 각 데이터셋별 클래스 수 정의. 
# CLIP 모델은 timm 라이브러리에서 불러오기 때문에 따로 가중치 경로 정의할 필요 없음.
WEIGHTS = {
    "cifar10": {
        "resnet50": "resnet50_cifar10.pth",
    },
    "cifar10c": {
        "resnet50": "resnet50_cifar10.pth",
    },
}

# 클래스 개수
classes = {
    "cifar10": 10,
    "cifar10c": 10,
    "cifar100": 100,
    "cifar100c": 100,
    "visda": 12,
    "imagenet": 1000,
    "imagenetc": 1000,
}


def return_base_model(
    name: str,
    device: device,
    dataset: str,
    path_to_weights: str = None,
    segments: Optional[int] = 0,
) -> Tuple[nn.Module, Callable]:
    if name.startswith("resnet"):   # Resnet 모델이면 timm 라이브러리에서 불러오기
        LOGGER.info(f"Loading Resnet version: {name}")
        model = timm.create_model(
            model_name=name,
            pretrained=dataset == "imagenet",
            num_classes=classes[dataset],
        )
        if dataset in ["cifar10", "cifar10c", "cifar100", "cifar100c"]:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  
            # CIFAR는 이미지 크기가 32x32라서 ResNet 모델의 첫 번째 convolution layer를 CIFAR에 맞게 수정해주기 (원래는 kernel_size=7, stride=2, padding=3)

        # Loading weights
        if path_to_weights is not None: # 가중치 경로 있으면 불러와서 로드 (ResNet 모델은 timm 라이브러리에서 불러오기 때문에 따로 가중치 경로 정의할 필요 없음, CIFAR용 ResNet 모델만 가중치 경로 정의해놓음)
            weights = torch.load(join(path_to_weights, WEIGHTS[dataset][name]))["state_dict"]
            model.load_state_dict(weights, strict=False)

        val_transform = transforms.Compose(
            [
                transforms.ToTensor(),  # 이미지 데이터를 텐서로 변환
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),   # CIFAR 데이터셋의 평균과 표준편차로 정규화?
            ]
            
        )
        '''
        코드상으론 dataset이 imagenet이어도 ResNet 모델이면 CIFAR용 전처리로 통일해서 쓰는 것 같은데..
        두 가능성이 있어
        1. 이 코드베이스가 ResNet을 사실상 CIFAR 중심으로만 진지하게 쓴다
        2. 아니면 ImageNet용 전처리 통일이 덜 정리된 연구 코드일 수 있다
        이건 나중에 실험 재현할 때 꼭 확인해야 할 포인트
        '''
        
        model.to(device)
        model.dtype = model.fc.weight.dtype

    elif name.startswith("clip"):   # CLIP 모델이면 CLIP 라이브러리에서 불러오기
        LOGGER.info(f"Loading CLIP version: {name[5:]}") # clip- 제거한 이름으로 CLIP 모델 버전 로깅
        model, val_transform = load_clip(name[5:], device=device, segments=segments) # CLIP 모델 불러오기 (CLIP 모델은 timm 라이브러리에서 불러오기 때문에 따로 가중치 경로 정의할 필요 없음)
        model.visual.dtype = model.visual.conv1.weight.dtype # CLIP 모델의 비전-텍스트 인코더 간의 타입 일치
    else:
        raise NotImplementedError

    return model, val_transform # 모델 + 검증용 전처리 함수(모델이 기대하는 입력 포맷)
