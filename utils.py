import numpy as np
import matplotlib.pyplot as plt
import models
import torch.backends.cudnn as cudnn


# =================================================================================================================
def get_module(name, *_args, **keywords):
	net = getattr(models, name)(*_args, **keywords)
	cudnn.benchmark = True

	return net


# ==================================================================================================
def plot_res(data, save_path):
	x_index = np.arange(len(data))
	plt.figure()
	plt.plot(x_index, data)
	plt.savefig(save_path, dpi=300)
