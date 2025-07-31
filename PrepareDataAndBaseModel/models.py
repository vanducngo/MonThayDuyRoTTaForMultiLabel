import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

def get_model_chexpert(cfg):
    return get_model(cfg, feature_extract=False, useWeight = True, numclasses=5)

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