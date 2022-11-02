import torchvision
from torchvision.transforms import ToTensor

torchvision.datasets.KMNIST(root="../data", train=True, transform=ToTensor(), download=True)
torchvision.datasets.KMNIST(root="../data", train=False, transform=ToTensor(), download=True)
