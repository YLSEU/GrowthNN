import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
import numpy as np

from utils import get_module
from datas.cifar10 import get_cifar10


if __name__ == '__main__':
	cnet_l1_output = []
	cnet_l2_output = []
	cnet_l3_output = []

	def make_hook_fn(out):
		def hook_fn(module, input, output):
				out.extend(output.detach().cpu().clone())

		return hook_fn

	net = get_module('CifarResNetBasicLC', [1, 1, 1], num_classes=10, image_channels=3).to('cuda')
	print(net)

	# =========================== 注册钩子函数 ==============
	net.layer1[0].cnet.register_forward_hook(make_hook_fn(cnet_l1_output))
	net.layer2[0].cnet.register_forward_hook(make_hook_fn(cnet_l2_output))
	net.layer3[0].cnet.register_forward_hook(make_hook_fn(cnet_l3_output))

	# ===============================================
	size = 32
	bs = 512
	lr = 0.001
	epochs = 50

	train_set, test_set = get_cifar10(size=size)
	train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
	test_loader = DataLoader(test_set, batch_size=bs, shuffle=False)

	optimizer = optim.Adam(net.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
	criterion = nn.CrossEntropyLoss()

	# ================== train ===============================
	for epoch in range(epochs):
		train_total = 0
		train_correct = 0
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

		print(f'Epoch {epoch}, Loss {train_loss / train_total:.5f}, lr: {optimizer.param_groups[0]["lr"]:.7f} Train Acc {train_correct / train_total * 100.0:.3f}%,', end=' ')
		scheduler.step()

		print(np.stack(cnet_l1_output, axis=0).mean(axis=0).shape, np.stack(cnet_l2_output, axis=0).mean(axis=0).shape, np.stack(cnet_l3_output, axis=0).mean(axis=0).shape)

	# ================== test ===============================
	net.eval()
	test_correct = 0
	test_total = 0

	with torch.no_grad():
		for step, (img, target) in enumerate(test_loader):
			test_total += img.size(0)

			img, target = img.to('cuda'), target.to('cuda')
			out = net(img)
			test_pred = out.argmax(dim=1, keepdim=True)
			test_correct += test_pred.eq(target.view_as(test_pred)).sum().item()

	print(f'Test Acc {test_correct / test_total * 100.0:.3f}%')
