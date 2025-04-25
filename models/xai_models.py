import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from models.inception1d import Inception1d

def get_state_dict(experiment_name, model):
	weights_path = f'../output/{experiment_name}/models/{model}/models/{model}.pth'
	return torch.load(weights_path)['model']

# pred is a tensor with gradients saved
def get_cam(pred, model, inp, class_id, n_ch=128):
	'''
	Display Grad-CAM heatmap from last layer of model.
	Model must be modified to have get_activations and get_activations_gradient
	'''
	pred[class_id].backward()
	
	activations = model.get_activations(inp).detach()
	gradients = model.get_activations_gradient()
	pooled_gradients = torch.mean(gradients, dim=[0,2])

	activations *= pooled_gradients.unsqueeze(0).unsqueeze(-1)
	heatmap = torch.mean(activations, dim=1).squeeze().cpu()

	heatmap = np.maximum(heatmap, 0)
	heatmap /= torch.max(heatmap)

	return heatmap

class Inception1dMod(nn.Module):
	def __init__(self, experiment_name):
		super().__init__()
		self.experiment_name = experiment_name
		self.model_name = 'fastai_inception1d'

		# get pretrained model
		m = Inception1d(
			num_classes=5, 
			input_channels=12, 
			use_residual=True,
			ps_head=0.5,
			lin_ftrs_head=[128],
			kernel_size=40
			)
		m.load_state_dict(get_state_dict(self.experiment_name, self.model_name))
		
		self.inception = m
		self.backbone = self.inception.layers[0]
		self.head = self.inception.layers[1]

		self.gradients = None

		del(m)
	
	def activations_hook(self, grad):
		self.gradients = grad
	
	def forward(self, x):
		x = self.backbone(x)
		h = x.register_hook(self.activations_hook)
		x = self.head(x)
		return x
	
	def get_activations_gradient(self):
		return self.gradients
	
	def get_activations(self, x):
		return self.backbone(x)