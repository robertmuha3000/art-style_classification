import torch
import torch.nn as nn

def create_model(num_classes: int, device: torch.device, pretrained: bool = True):
    """
    Load ResNet34 from the same hub tag you used, replace the final layer,
    and move to device.
    """
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    return model

def freeze_backbone_keep_fc_trainable(model):
    """
    Freeze all params, unfreeze only the final fc layer (head training phase).
    """
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

def unfreeze_layer3_layer4_and_fc(model):
    """
    Unfreeze layer4, layer3, and fc (fine-tuning phase), keep others frozen.
    """
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.layer3.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True

def make_param_groups(model, lr_backbone: float = 1e-4, lr_head: float = 3e-4):
    """
    Return optimizer param groups exactly like your script (layer4, layer3, fc).
    """
    pg = [
        {"params": [p for p in model.layer4.parameters() if p.requires_grad], "lr": lr_backbone},
        {"params": [p for p in model.layer3.parameters() if p.requires_grad], "lr": lr_backbone},
        {"params": [p for p in model.fc.parameters()     if p.requires_grad], "lr": lr_head},
    ]
    return pg

def load_state(model, checkpoint_path: str, device: torch.device, strict: bool = True):
    """
    Load a state_dict saved with torch.save(model.state_dict(), ...).
    """
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=strict)
    return model
