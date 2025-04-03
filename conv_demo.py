import torch
import torch.nn as nn

# 设置随机种子以确保结果可重复
torch.manual_seed(0)

# 定义输入张量 (batch_size=1, in_channels=3, height=4, width=4)
input_tensor = torch.tensor([
	[
		[[1, 2, 3, 4],
		 [5, 6, 7, 8],
		 [9, 10, 11, 12],
		 [13, 14, 15, 16]
		 ],

		[[17, 18, 19, 20],
		 [21, 22, 23, 24],
		 [25, 26, 27, 28],
		 [29, 30, 31, 32]],

		[[33, 34, 35, 36],
		 [37, 38, 39, 40],
		 [41, 42, 43, 44],
		 [45, 46, 47, 48]]
	]
], dtype=torch.float32)

# 定义卷积核 (out_channels=2, in_channels=3, kernel_height=3, kernel_width=3)
kernel_manual = torch.tensor([
	[
		[[1, 0, -1],
		 [1, 0, -1],
		 [1, 0, -1]],

		[[1, 1, 1],
		 [0, 0, 0],
		 [-1, -1, -1]],

		[[-1, 0, 1],
		 [-2, 0, 2],
		 [-1, 0, 1]]
	],
	[
		[[-1, -1, -1],
		 [0, 0, 0],
		 [1, 1, 1]],

		[[-1, 0, 1],
		 [-2, 0, 2],
		 [-1, 0, 1]],

		[[-1, 2, -1],
		 [-2, 4, -2],
		 [-1, 2, -1]]
	]
], dtype=torch.float32)


# 手动计算卷积
def manual_conv2d(input_tensor, kernel):
	batch_size, in_channels, in_height, in_width = input_tensor.shape
	out_channels, _, kernel_height, kernel_width = kernel.shape

	# 计算输出尺寸
	out_height = in_height - kernel_height + 1
	out_width = in_width - kernel_width + 1

	# 初始化输出张量
	output = torch.zeros(batch_size, out_channels, out_height, out_width)

	# 进行卷积操作
	for b in range(batch_size):
		for oc in range(out_channels):
			for ic in range(in_channels):
				for oh in range(out_height):
					for ow in range(out_width):
						h_start = oh
						h_end = h_start + kernel_height
						w_start = ow
						w_end = w_start + kernel_width
						input_slice = input_tensor[b, ic, h_start:h_end, w_start:w_end]
						output[b, oc, oh, ow] += torch.sum(input_slice * kernel[oc, ic])
	return output


# 手动卷积结果
manual_output = manual_conv2d(input_tensor, kernel_manual)
print("手动卷积输出:")
print(manual_output)

# 使用 PyTorch 的 nn.Conv2d 进行卷积
conv_layer = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)
# 将手动定义的卷积核赋值给 Conv2d 层的权重
conv_layer.weight.data = kernel_manual

# PyTorch 卷积结果
torch_output = conv_layer(input_tensor)
print("PyTorch Conv2d 输出: ")
print(torch_output)

# 验证两者是否相等（考虑浮点数精度）
print("两者是否相等: ", torch.allclose(manual_output, torch_output, atol=1e-5))
