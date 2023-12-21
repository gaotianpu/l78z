import torch
from torchvision import datasets, models, transforms
from torchvision.transforms import v2,ToTensor

H, W = 32, 32
img = torch.randint(0, 256, size=(3, H, W), dtype=torch.uint8)

print(img)

# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

# t = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
# print(t(img))


# https://pytorch.org/vision/stable/transforms.html
t = v2.Compose([
    v2.ToDtype(torch.float32) #, scale=True
])
print(t(img))


t = v2.Compose([
    v2.ToDtype(torch.float32),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
print(t(img))