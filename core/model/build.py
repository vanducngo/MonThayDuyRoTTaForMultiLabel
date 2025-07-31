import os
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

import torch
import torch.nn as nn
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model

def get_model(cfg, feature_extract=False, useWeight=True, numclasses=5):
    model = None
    arch = cfg.MODEL.ARCH.lower()
    
    print(f">>> Loading model: {arch} | useWeight: {useWeight} | num_classes: {numclasses}")
    
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if useWeight else None
    model = mobilenet_v3_small(weights=weights)

    if hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            model.classifier = nn.Sequential(
                nn.Linear(model.classifier[0].in_features, 512),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(512, numclasses)
            )
        else:
            raise TypeError(f"Unsupported classifier type: {type(model.classifier)}")
    else:
        raise AttributeError("Model does not have 'fc' or 'classifier' attribute.")

    print(f"Model pre-trained on ImageNet loaded.")
    if feature_extract:
        print("Feature extracting mode: All layers frozen except the final classifier.")
    else:
        print("Fine-tuning mode: All layers are trainable.")
        
    print(f"Model adapted for {numclasses} classes.")
    
    return model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_pretrained_model(cfg):
    model_path = './ckpt/path.pth'
    print(f"Loading fine-tuned weights from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run the training script first.")
    
    print(f"Found fine-tuned model at {model_path}")
    # Load the pre-trained model architecture
    model = get_model(cfg, feature_extract=False, useWeight = True, numclasses=5)
    # Load the fine-tuned weights
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    print(f"Loaded fine-tuned model from {model_path}")
    
    print("Fine-tuned model loaded successfully.")
    return model

def build_model(cfg):
    dataset_name = cfg.DATASET.NAME

    if dataset_name in ["cifar10", "cifar100"]:
        # --- Logic gốc cho CIFAR ---
        print(f"Building pre-trained model for {dataset_name}...")
        base_model = load_model(
            cfg.MODEL.ARCH, 
            cfg.CKPT_DIR,
            dataset_name, 
            ThreatModel.corruptions
        ).cpu()

    elif 'CXR' in dataset_name:
        base_model = get_pretrained_model(cfg)
    else:
        raise NotImplementedError(f"Model building logic not implemented for dataset: {dataset_name}")

    return base_model