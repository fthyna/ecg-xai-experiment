import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from fastai.layers import *
from fastai.core import *

from models.basic_conv1d import AdaptiveConcatPool1d,create_head1d

########################################################################################################
# Inception time inspired by https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py and https://github.com/tcapelle/TimeSeries_fastai/blob/master/inception.py

def conv(in_planes, out_planes, kernel_size=3, stride=1):
	"""
	Helper constructor. Constructs a 1D convolutional layer with automatic padding.

	The padding is set to `(kernel_size-1) // 2`, which preserves input length
	only when stride is 1 and kernel_size is odd.

	Args:
		in_planes (int): Number of input channels.
		out_planes (int): Number of output channels.
		kernel_size (int, optional): Size of the convolutional kernel. Must be odd. Defaults to 3.
		stride (int, optional): Stride of the convolution. Defaults to 1.

	Returns:
		nn.Conv1d: A 1D convolutional layer with specified parameters.
	"""

	return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
					padding=(kernel_size-1)//2, bias=False)

class NoOp(nn.Module):
	"""
	No-operator module which returns the input unchanged.

	Used to disable bottleneck layer in class InceptionBlock1d when not needed.
	"""

	# No __init__ required, automatically inherits parent.

	def forward(self, x):
		return x

class InceptionBlock1d(nn.Module):
	"""
	Inception block module.
	"""
	def __init__(self, ni, nb_filters, kss, stride=1, act='linear', bottleneck_size=32):
		"""
		Initialize an InceptionBlock1d NN module.

		The block consists of:
		* a bottleneck which modifies the input's channel count,
		* convolution branches with specified kernel sizes as well as a max pool branch,
		* batch normalization and relu activation applied to the concatenated branch outputs.
		
		The final output tensor has (len(kss)+1) * nb_filters) channels.

		Args:
			ni (int): Number of input channels.
			nb_filters (int): Number of output filters/channels for each convolution branch.
			kss (list): List of kernel sizes for the inception logic.
			act (string): Unused in this implementation. Might be a placeholder for activation function selection.
			bottleneck_size (int): If positive, a 1d conv bottleneck is applied to reduce ni to bottleneck_size.
		"""

		# Initialize the parent module
		super().__init__()

		# If bottleneck_size > 0, create bottleneck module which reduces the num of input channels (ni) to bottleneck_size through 1d conv
		self.bottleneck = conv(ni, bottleneck_size, 1, stride) if (bottleneck_size>0) else NoOp()

		# Main conv branches. Creates a list of 1d conv modules, one for each kernel size
		self.convs = nn.ModuleList([conv(bottleneck_size if (bottleneck_size>0) else ni, nb_filters, ks) for ks in kss])

		# Max pooling branch. Applies 1D max pooling followed by convolution with kernel size 1. Bypasses bottleneck (uses raw input).
		self.conv_bottle = nn.Sequential(nn.MaxPool1d(3, stride, padding=1), conv(ni, nb_filters, 1))

		# Batch normalization + ReLU activation for post-processing
		# Batch normalization = normalize values across every channel (all values on the same batch/sequence length plane normalized by the same factor)
		self.bn_relu = nn.Sequential(nn.BatchNorm1d((len(kss)+1)*nb_filters), nn.ReLU())

	def forward(self, x):
		#print("block in",x.size())

		# Apply bottleneck if bottleneck_size is positive. Otherwise bottled = input
		bottled = self.bottleneck(x)

		# Create all branches (self.convs(bottled), conv_bottle()), concatenate across channels (torch.cat), then normalize and activate
		out = self.bn_relu(torch.cat([c(bottled) for c in self.convs]+[self.conv_bottle(x)], dim=1))
		return out

class Shortcut1d(nn.Module):
	def __init__(self, ni, nf):
		"""
		Initialize a 1d shortcut NN module.

		Args:
			ni (int): Number of input channels.
			nf (int): Number of output channels.
		"""

		super().__init__()

		# Set the activation function to ReLU
		self.act_fn=nn.ReLU(True)

		# Add 1d convolution with ni = input chs, nf = output chs, and kernel size = 1
		self.conv=conv(ni, nf, 1)

		# Add batch normalization for the resulting convolution output.
		self.bn=nn.BatchNorm1d(nf)

	def forward(self, inp, out): 

		#print("sk",out.size(), inp.size(), self.conv(inp).size(), self.bn(self.conv(inp)).size)
		#input()

		# Conv the input as specified, batch normalize, add it to output, then activate
		return self.act_fn(out + self.bn(self.conv(inp)))
		
class InceptionBackbone(nn.Module):
	def __init__(self, input_channels, kss, depth, bottleneck_size, nb_filters, use_residual):
		"""
		Initialize the backbone of inception time.

		Args:
			input_channels (int): Number of input channels.
			kss (int): List of kernel sizes for each inception block.
			depth (int): Number of inception blocks. Must be divisible by 3 to allow skip connections.
			bottleneck_size (int): Bottleneck size applied to each inception block.
			nb_filters (int): Number of output filters for each branch in each inception block.
			use_residual (bool): Enables skip connections every 3 blocks.

		The first block accepts input_channels while the rest accept the output of the previous block.
		Each block outputs (len(kss) + 1) * nb_filters channels, which is also the final output channels.
		The final output would be n * ((len(kss) + 1) * nb_filters) * length_of_time_data.
		"""

		super().__init__()

		self.depth = depth
		assert((depth % 3) == 0)
		self.use_residual = use_residual

		n_ks = len(kss) + 1
		self.im = nn.ModuleList([InceptionBlock1d(input_channels if d==0 else n_ks*nb_filters,nb_filters=nb_filters,kss=kss, bottleneck_size=bottleneck_size) for d in range(depth)])
		self.sk = nn.ModuleList([Shortcut1d(input_channels if d==0 else n_ks*nb_filters, n_ks*nb_filters) for d in range(depth//3)])
		
	def forward(self, x):
		"""
		Pass input through a sequence of inception blocks.

		If use_residual is true, every triplet of consecutive blocks applies a skip connection,
		adding the triplet's input (tracked in input_res) to the triplet's output.
		input_res is updated after every skip connection.
		"""

		input_res = x
		for d in range(self.depth):
			x = self.im[d](x)
			if self.use_residual and d % 3 == 2:
				x = (self.sk[d//3])(input_res, x)
				input_res = x.clone()
		return x

class Inception1d(nn.Module):
	'''inception time architecture'''
	def __init__(self, num_classes=2, input_channels=8, kernel_size=40, depth=6, bottleneck_size=32, nb_filters=32, use_residual=True,lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True):
		"""
		Construct a complete inception time module.
		
		Args:
			num_classes (int): Number of classes for classification. Affects the head layer.
			input_channels (int): Number of input channels
			kernel_size (int): The maximum kernel size. 3 kernels will be created with sizes: kernel_size, kernel_size//2, kernel_size//4.
			depth (int): Number of inception blocks
			bottleneck_size (int): Bottleneck size applied to each inception block.
			nb_filters (int): Number of output filters for each branch in each inception block.
			use_residual (bool): Enables skip connections every 3 blocks.
			lin_ftrs_head ()
		"""

		super().__init__()
		assert(kernel_size>=40)
		kernel_size = [k-1 if k%2==0 else k for k in [kernel_size,kernel_size//2,kernel_size//4]] #was 39,19,9
		
		layers = [InceptionBackbone(input_channels=input_channels, kss=kernel_size, depth=depth, bottleneck_size=bottleneck_size, nb_filters=nb_filters, use_residual=use_residual)]
	
		n_ks = len(kernel_size) + 1
		#head
		head = create_head1d(n_ks*nb_filters, nc=num_classes, lin_ftrs=lin_ftrs_head, ps=ps_head, bn_final=bn_final_head, bn=bn_head, act=act_head, concat_pooling=concat_pooling)
		layers.append(head)
		#layers.append(AdaptiveConcatPool1d())
		#layers.append(Flatten())
		#layers.append(nn.Linear(2*n_ks*nb_filters, num_classes))
		self.layers = nn.Sequential(*layers)

	def forward(self,x):
		return self.layers(x)
	
	def get_layer_groups(self):
		depth = self.layers[0].depth
		if(depth>3):
			return ((self.layers[0].im[3:],self.layers[0].sk[1:]),self.layers[-1])
		else:
			return (self.layers[-1])
	
	def get_output_layer(self):
		return self.layers[-1][-1]
	
	def set_output_layer(self,x):
		self.layers[-1][-1] = x
	
def inception1d(**kwargs):
	"""Constructs an Inception model
	"""
	return Inception1d(**kwargs)
