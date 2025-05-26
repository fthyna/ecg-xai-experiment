# DO NOT USE
# THE CONTENTS OF THIS FILE HAVE BEEN MOVED TO `dataloader.py`

from dataloader import PTBXL
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import helper
import pickle
import os

from sklearn.preprocessing import StandardScaler, LabelBinarizer, MultiLabelBinarizer

class PTBXL_Preprocessor():
	def __init__(self, task='mi_detection'):
		'''Initialize PTB-XL preprocessor. All parameters are fixed.'''

		# fitted dataset
		print("Initializing fitting dataset")
		self.dataset = PTBXL(sampling_frequency=500, subset='train', task=task)
		self.batch_size = 128
		self.loader = DataLoader(self.dataset, batch_size=self.batch_size, num_workers=4)

		# print("Dataset sample:")
		# print(self.loader.dataset[0])

		# general attributes
		self.n_channels = 12
		self.sampling_frequency = 500
		self.subset = 'train'
		self.n_records = len(self.dataset)
		self.classes = self.dataset.classes
		self.n_classes = len(self.classes)

		# computed attributes
		self.ch_means = None
		self.ch_stds = None

	def fit(self):
		# standard scaler
		ch_sums = np.zeros(self.n_channels)
		ch_sums_sq = np.zeros(self.n_channels)
		n_total = 0

		for ecgs, _ in tqdm(self.loader, desc='Computing normalization parameters'):
			ecgs = ecgs.cpu()
			ch_sums += ecgs.sum(dim=(0,2)).numpy()
			ch_sums_sq += (ecgs ** 2).sum(dim=(0,2)).numpy()
			n_total += ecgs.shape[0] * ecgs.shape[2]

		mean = ch_sums/n_total
		std = np.sqrt((ch_sums_sq / n_total) - (mean ** 2))

		# label binarizer
		lb = None
		if (len(self.classes) == 2):
			lb = LabelBinarizer()
			lb.fit(self.classes)
		
		# multilabel binarizer
		mlb = MultiLabelBinarizer()
		tmp_class = [[c] for c in self.classes]
		mlb.fit(tmp_class)

		self.classes = mlb.classes_ # ensure classes attr consistent with binarizers
		self.ch_means = mean
		self.ch_stds = std
		self.bin_binarizer = lb
		self.multi_binarizer = mlb
	
	def normalize(self, ecgs):
		single = False
		if len(ecgs.shape) == 2:
			single = True
			ecgs = ecgs.unsqueeze(0)
		assert ecgs.shape[1] == self.n_channels, f'Expected {self.n_channels} leads, got {ecgs.shape[1]}'

		# convert everything to tensor at the correct device with the correct batch dimension
		device = ecgs.device
		mean_tnsr = torch.tensor(self.ch_means, device=device)[None, :, None]
		std_tnsr = torch.tensor(self.ch_stds, device=device)[None, :, None]
		
		ret = (ecgs - mean_tnsr) / std_tnsr
		if single:
			ret = ret.squeeze(0)
		return ret
	
	def binarize_labels(self, labels, multi=True):
		'''
		Args:
			labels (array of list of str): The labels to transform
			multi (bool): Applies MultiLabelBinarizer if true, otherwise applies LabelBinarizer
		Returns:
			np.ndarray: The transformed labels
		'''
		if not isinstance(labels, np.ndarray):
			labels = np.array(labels)
		if multi:
			return torch.tensor(self.multi_binarizer.transform(labels))
		else:
			labels = np.array([x[0] for x in labels])
			return torch.tensor(self.bin_binarizer.transform(labels))
		
	def save(self, file_name, folder_path='../myoutput/preprocessor/'):
		'''`.pkl` extension added automatically'''
		helper.check_create_folder(folder_path)
		with open(os.path.join(folder_path, file_name+'.pkl'), 'wb') as f:
			pickle.dump(self.__dict__, f)
	
	def load(self, file_name, folder_path='../myoutput/preprocessor/'):
		'''`.pkl` extension added automatically'''
		with open(os.path.join(folder_path, file_name+'.pkl'), 'rb') as f:
			attrs = pickle.load(f)
		self.__dict__.update(attrs)