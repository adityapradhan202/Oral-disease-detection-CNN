"""Trains a PyTorch image classification model using device-agnostic code."""

import os
import torch
from torchvision import transforms


# Not using going_modular.script_name because the train.py will have access to these...
import data_setup, engine, model_builder, utils

# Setup some hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Setup directories
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Class to avoid bottleneck issue during training
# class GPUAugmentTransform:
#     def __init__(self):
#         self.resize_on_cpu = transforms.Resize(size=(64,64))
#     def __call__(self, image):
#         image = self.resize_on_cpu(image)
#         image = transforms.ToTensor()(image)
#         return image
      
# Create transforms
data_transform = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.ToTensor()
])

# Create dataloaders and get class_names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir, test_dir=test_dir,
    transform=data_transform, batch_size=BATCH_SIZE,
)

# Create the model
model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device=device)

# Setup loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

from timeit import default_timer as timer

start_time = timer()
# Train the model...
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn, optimizer=optimizer,
             epochs=NUM_EPOCHS, device=device)

end_time = timer()
print(f"Total training time: {(end_time - start_time):.3f} seconds")

# Save the model to file
utils.save_model(
    model=model,
    target_dir="models",
    model_name="05_going_modular_script_tinyvgg_model.pth"
)
