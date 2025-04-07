import warnings

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F


# =================================================================================================================
class LNet(nn.Module):
	def __init__(self, in_channels):
		super(LNet, self).__init__()

		self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Linear(in_channels, 1)

	def forward(self, x):
		x = self.global_avg_pool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		x = self.relu1(x)

		return x

	def relu1(self, x):
		return torch.clamp(x, 0, 1)


# =================================================================================================================
class CNet(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(CNet, self).__init__()

		self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Linear(in_channels, out_channels)

	def forward(self, x):
		x = self.global_avg_pool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		x = self.relu1(x)

		return x

	def relu1(self, x):
		return torch.clamp(x, 0, 1)

# =================================================================================================================
class BasicBlockLC(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1):
		super(BasicBlockLC, self).__init__()

		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.cnet = CNet(in_planes, planes)

		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.lnet = LNet(in_planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion * planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion * planes)
			)

	def forward(self, x):
		channel_impotrance_score = self.cnet(x)  # [bs, 16, 32, 32] ---> [bs, 16]
		channel_impotrance_score = channel_impotrance_score.view(channel_impotrance_score.size(0), -1, 1, 1)  # [bs, 16] ---> [bs, 16, 1, 1]

		out = F.relu(self.bn1(self.conv1(x)))
		out = out * channel_impotrance_score

		out = self.bn2(self.conv2(out))  # [bs, 16, 32, 32]
		layer_importance_score = self.lnet(x)  # [bs, 1]
		layer_importance_score = layer_importance_score.view(-1, 1, 1, 1)
		out = out * layer_importance_score

		out += self.shortcut(x)

		out = F.relu(out)

		return out


# =================================================================================================================
class BottleneckLC(nn.Module):
	expansion = 4

	def __init__(self, in_planes, planes, stride=1):
		super(BottleneckLC, self).__init__()

		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)

		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.cnet = CNet(in_planes, planes)

		self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(self.expansion * planes)
		self.lnet = LNet(in_planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion * planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion * planes)
			)

	def forward(self, x):
		channel_impotrance_score = self.cnet(x)
		channel_impotrance_score = channel_impotrance_score.view(channel_impotrance_score.size(0), -1, 1, 1)

		out = F.relu(self.bn1(self.conv1(x)))

		out = F.relu(self.bn2(self.conv2(out)))
		out = out * channel_impotrance_score

		out = self.bn3(self.conv3(out))

		layer_importance_score = self.lnet(x)
		layer_importance_score = layer_importance_score.view(-1, 1, 1, 1)
		out = out * layer_importance_score

		out += self.shortcut(x)

		out = F.relu(out)

		return out


# =================================================================================================================
class CifarResNet(nn.Module):
	def __init__(self, block, num_blocks, num_classes=10, image_channels=3, batchnorm=True):
		super(CifarResNet, self).__init__()

		self.in_planes = 16

		if batchnorm:
			self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
			self.bn1 = nn.BatchNorm2d(16)
		else:
			self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=3, stride=1, padding=1, bias=True)
			self.bn1 = nn.Sequential()

		self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
		self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

		self.linear = nn.Linear(64 * block.expansion, num_classes)

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1] * (num_blocks - 1)
		layers = []

		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion

		return nn.Sequential(*layers)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))

		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)

		out = F.avg_pool2d(out, 8)
		out = out.view(out.size(0), -1)
		out = self.linear(out)

		return out


# =================================================================================================================
def CifarResNetBasicLC(num_blocks, num_classes=10, image_channels=3):
	assert len(num_blocks) == 3, "3 blocks are needed, but %d is found." % len(num_blocks)

	print('num_classes=%d, image_channels=%d' % (num_classes, image_channels))

	return CifarResNet(BasicBlockLC, num_blocks, num_classes=num_classes, image_channels=image_channels)


# =================================================================================================================
def CifarResNetBottleLC(num_blocks, num_classes=10, image_channels=3):
	assert len(num_blocks) == 3, "3 blocks are needed, but %d is found." % len(num_blocks)

	print('num_classes=%d, image_channels=%d' % (num_classes, image_channels))

	return CifarResNet(BottleneckLC, num_blocks, num_classes=num_classes, image_channels=image_channels)
