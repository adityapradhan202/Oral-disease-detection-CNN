import torch
import torchvision
from torch import nn
from pathlib import Path
from PIL import Image
from torchvision import transforms
import os
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

model_save_path = Path('model/eff2_model97.pth')
class_names = ['Caries', 'Gingivitis', 'Hypodontia', 'Mouth_ulcer', 'Tooth_discoloration']

loaded_eff2_weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
loaded_eff2_model = torchvision.models.efficientnet_b0(
    weights=loaded_eff2_weights
).to(device=device)

for param in loaded_eff2_model.features.parameters():
    param.requires_grad = False

for param in loaded_eff2_model.features[-1].parameters():
    param.requires_grad = True

loaded_eff2_model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=5, bias=True)
).to(device=device)

loaded_eff2_dict = torch.load(f=model_save_path, weights_only=True)
loaded_eff2_model.load_state_dict(loaded_eff2_dict)

def predict_image(
                  img_path:str, 
                  model_transform:transforms=loaded_eff2_weights.transforms(),
                  device:torch.device=device, model:nn.Module=loaded_eff2_model):
    """A function to make predictions on custom images downloaded from the internet."""

    image_tensor = model_transform(Image.open(img_path))
    image_tensor = image_tensor.unsqueeze(dim=0)
    
    model.to(device=device)
    model.eval()
    with torch.inference_mode():
        logit = model(image_tensor.to(device))
        print(class_names[torch.softmax(logit, dim=1).argmax(dim=1).item()])

if __name__ == "__main__":
    predict_image(
        model=loaded_eff2_model,
        img_path='sample_images/ulcer_m.jpg',
        device=device,
        model_transform=loaded_eff2_weights.transforms()
    )