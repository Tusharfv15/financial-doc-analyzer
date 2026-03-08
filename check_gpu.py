import torch

device = torch.cuda.is_available()
print(device)