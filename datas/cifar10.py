from torchvision import datasets, transforms
from .cutout import Cutout


def get_cifar10(size):
	train_transform = transforms.Compose(
		[
			transforms.RandomCrop(size, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
			Cutout(n_holes=1, length=16)
		])

	test_transform = transforms.Compose(
		[transforms.ToTensor(),
		 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

	train_dataset = datasets.CIFAR10(root=r'D:\Dataset\public\cifar10', train=True, download=True, transform=train_transform)
	test_dataset = datasets.CIFAR10(root=r'D:\Dataset\public\cifar10', train=False, download=True, transform=test_transform)
	input_shape = 32 * 32 * 3
	num_class = 10

	return train_dataset, test_dataset
