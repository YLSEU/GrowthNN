import torch

from utils import get_module
import models
from datas.cifar10 import get_data
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn


print(getattr(models, 'CifarResNetBottleLC'))
net = get_module('CifarResNetBottleLC', [6, 6, 6], num_classes=10, image_channels=3).to('cuda')
print(net)

size = 32
bs = 512
lr = 0.001
epochs = 50

train_set, test_set = get_data(size=size)
train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
test_loader = DataLoader(test_set, batch_size=bs, shuffle=False)

optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
	train_total = 0
	train_correct = 0
	test_total = 0
	test_correct = 0
	train_loss = 0

	net.train()
	for step, (img, target) in enumerate(train_loader):
		train_total += img.size(0)

		img, target = img.to('cuda'), target.to('cuda')
		out = net(img)

		optimizer.zero_grad()
		loss = criterion(out, target)
		loss.backward()
		optimizer.step()
		train_loss += loss.item()

		train_pred = out.argmax(dim=1, keepdim=True)
		train_correct += train_pred.eq(target.view_as(train_pred)).sum().item()

	print(f'Epoch {epoch}, Loss {train_loss / train_total:.5f}, Train Acc {train_correct / train_total * 100.0:.3f}%,', end=' ')

	net.eval()
	with torch.no_grad():
		for step, (img, target) in enumerate(test_loader):
			test_total += img.size(0)

			img, target = img.to('cuda'), target.to('cuda')
			out = net(img)
			test_pred = out.argmax(dim=1, keepdim=True)
			test_correct += test_pred.eq(target.view_as(test_pred)).sum().item()

	print(f'Test Acc {test_correct / test_total * 100.0:.3f}%')
	scheduler.step()
