import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision
from pathlib import Path

# Device agnostic code
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Current device: {device}")

eff2_weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
eff2_transform = eff2_weights.transforms()

eff2_model = torchvision.models.efficientnet_b0(weights=eff2_weights).to(device=device)
# Freezing all the base layers
for param in eff2_model.features.parameters():
    param.requires_grad = False

# Unfreezing parameters of the last feature layer
for param in eff2_model.features[-1].parameters():
    param.requires_grad = True

# Changing classification layer according to our own problem
eff2_model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=5, bias=True)
).to(device=device)

# Creating data loaders
image_folder = Path("dental_disease_split_dataset")
train_dir = image_folder / "train"
test_dir = image_folder / "test"

train_datav2 = ImageFolder(
    root=train_dir,
    transform=eff2_transform,
    target_transform=None
)

test_datav2 = ImageFolder(
    root=test_dir,
    transform=eff2_transform,
    target_transform=None
)

train_dataloaderV2 = DataLoader(
    dataset=train_datav2,
    batch_size=32,
    num_workers=0,
    shuffle=True,
    pin_memory=True)

test_dataloaderV2 = DataLoader(
    dataset=test_datav2,
    batch_size=32,
    num_workers=0,
    shuffle=False,
    pin_memory=True)
class_names = train_datav2.classes

# Loss function and optimizer
loss_fnV2 = nn.CrossEntropyLoss()
optimizerV2 = torch.optim.Adam(params=eff2_model.parameters(), lr=0.001)

from torchscript import engine
from timeit import default_timer as time_stamp

# Set manual seed before training part.
torch.manual_seed(42)
torch.cuda.manual_seed(42)

start_time_effnet = time_stamp()

results_effnet2 = engine.train(
    model=eff2_model,
    train_dataloader=train_dataloaderV2,
    test_dataloader=test_dataloaderV2,
    loss_fn=loss_fnV2,
    optimizer=optimizerV2,
    epochs=5,
    device=device
)

end_time_effnet = time_stamp()
total_time_effnet = end_time_effnet - start_time_effnet
print(f"Total training time: {total_time_effnet}")

# Code for saving the model
eff2_save_path = "models/eff2_model97.pth"
torch.save(obj=eff2_model.state_dict(), f=eff2_save_path)
print(f"Model has been saved!")
