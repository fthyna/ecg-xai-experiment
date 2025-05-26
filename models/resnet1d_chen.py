from models.resnet1d import *
import torch.nn as nn
import torch.nn.functional as F

class ResNet1d_chen(nn.Sequential):
	def __init__(self, input_channels=12, inplanes=64, kernel_size=15, num_classes=5):
		self.inplanes = inplanes
		layers = []

		# stem
		print("creating stem")
		layers.append(conv(input_channels, inplanes, kernel_size=kernel_size))
		layers.append(nn.BatchNorm1d(inplanes))
		layers.append(nn.ReLU(inplace=False))
		layers.append(nn.Dropout(0.5, inplace=False))

		# backbone
		print("creating backbone")
		for i in range(4):
			ch_i = max(inplanes, inplanes*(i))
			ch_o = inplanes*(i+1)
			downsample = nn.Sequential(
				nn.MaxPool1d(kernel_size=4, stride=4),
				conv(ch_i, ch_o, kernel_size=1)
			)
			layers.append(BasicBlock1d(ch_i, ch_o, kernel_size=[kernel_size, kernel_size], stride=4, drop_p=0.5, downsample=downsample))
		
		# head
		print("creating head")
		layers.append(nn.AdaptiveAvgPool1d(1))
		layers.append(nn.Flatten())
		layers.append(nn.Linear(inplanes*4, num_classes))

		super().__init__(*layers)

class ResNet1d_chen_100hz(nn.Sequential):
	def __init__(self, input_channels=12, inplanes=64, kernel_size=5, num_classes=5, drop_p=0.5):
		self.inplanes = inplanes
		layers = []

		# stem
		print("creating stem")
		layers.append(conv(input_channels, inplanes, kernel_size=kernel_size))
		layers.append(nn.BatchNorm1d(inplanes))
		layers.append(nn.ReLU(inplace=False))
		layers.append(nn.Dropout(drop_p, inplace=False))

		# backbone
		print("creating backbone")
		for i in range(4):
			ch_i = max(inplanes, inplanes*(i))
			ch_o = inplanes*(i+1)
			downsample = nn.Sequential(
				nn.ConstantPad1d((0, 1), 0),
				nn.MaxPool1d(kernel_size=2, stride=2),
				conv(ch_i, ch_o, kernel_size=1)
			)
			layers.append(BasicBlock1d(ch_i, ch_o, kernel_size=[kernel_size, kernel_size], stride=2, drop_p=drop_p, downsample=downsample))
		
		# head
		print("creating head")
		layers.append(nn.AdaptiveAvgPool1d(1))
		layers.append(nn.Flatten())
		layers.append(nn.Linear(inplanes*4, num_classes))
		layers.append(nn.Sigmoid())

		super().__init__(*layers)

		