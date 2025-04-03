import torch
import torch.nn as nn
import torch.nn.functional as F


class LNet(nn.Module):
    def __init__(self, in_channels):
        super(LNet, self).__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, 1)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.relu1(x)

        return x


class CNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNet, self).__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, out_channels)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.relu1(x)

        return x


class DynamicConv(nn.Module):
    def __init__(self, conv, in_channels, out_channels):
        super(DynamicConv, self).__init__()

        self.conv = conv
        self.cnet = CNet(in_channels, out_channels)

    def forward(self, x):
        conv_output = self.conv(x)
        salience_scores = self.cnet(x)
        salience_scores = salience_scores.view(salience_scores.size(0), -1, 1, 1)

        return conv_output * salience_scores


class DynamicResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DynamicResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dynamic_conv1 = DynamicConv(self.conv1, in_channels, out_channels)
        self.dynamic_conv2 = DynamicConv(self.conv2, out_channels, out_channels)
        self.lnet = LNet(in_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.dynamic_conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dynamic_conv2(out)
        out = self.bn2(out)

        salience_score = self.lnet(x)

        if salience_score.item() == 0:
            return residual
        else:
            out = out * salience_score + residual
            out = self.relu(out)

            return out


class LCNetResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(LCNetResNet, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def lcnet_resnet18(pretrained=False, **kwargs):
    model = LCNetResNet(DynamicResNetBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        # 加载预训练权重（如果需要）
        pass
    return model

# 示例：创建一个 LC-Net + ResNet-18 模型
model = lcnet_resnet18(num_classes=10)
print(model)
