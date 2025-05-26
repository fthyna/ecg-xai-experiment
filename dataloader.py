import pandas as pd
import numpy as np
import wfdb
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import tensor
import ast
import utils
from ecg_processing import resize_ecg
import helper
from tqdm import tqdm
import pickle
import os

class PTBXL(Dataset):
	'''
	Attributes:
		root_path (str): Path to dataset root folder
		sampling_frequency (int): Dataset sampling frequency
		subset (str): Dataset subset ('train', 'val' or 'test')
		task (str): Classification task, affects labels and data slice ('superdiagnostic', 'mi_detection' or 'mi_norm_detection')
		
		paths (list of str): Paths to raw ECG signals
		meta (pd.DataFrame): DataFrame containing metadata of each record
		labels (np.ndarray of list of str): Array containing true labels
		classes (list of str): (C,) list containing class names corresponding to labels
	'''
	def __init__(self, root_path='../data/ptbxl/', sampling_frequency:int=100, subset=None, task='mi_detection', resize_len=None, mode='raw', preprocessor_code=None, debug=False):
		'''
		Initialize the dataset.

		The dataset slice to load depends on the data subset and the task (multilabel or MI classification)

		Args:
			root_path (str): The path containing the dataset (ptbxl_database.csv, records folders, and SCP descriptor scp_statements.csv)
			sampling_frequency (int): The sampling frequency being used (100/500 Hz)
			subset (str or None): The subset of data being loaded ('train', 'val', 'test', or None for all)
			task (str): The classification task, should be either: superdiagnostic, mi_detection or mi_norm_detection
			resize_len (int or None): Length in timesteps to resize the records into
			mode (str): 'raw' loads the dataset without preprocessing, 'prepped' loads it with preprocessing.
			preprocessor_code (str or None): Name to the fitted preprocessor to use
			debug (bool): If set to True, __getitem__ will also return the item's index as the 3rd tuple element
		'''

		if sampling_frequency not in [100, 500]:
			raise ValueError("sampling_frequency must be 100 or 500")
		if task not in ["superdiagnostic", "mi_detection", "mi_norm_detection"]:
			raise ValueError("Invalid task")

		self.root_path = root_path
		self.sampling_frequency = sampling_frequency
		self.subset = subset
		self.task = task
		self.shape = (12, sampling_frequency*100)
		self.debug = debug

		# read CSV containing data
		df = pd.read_csv(root_path+'ptbxl_database.csv', index_col='ecg_id', na_values=[""])
		df['scp_codes'] = df['scp_codes'].apply(lambda x: ast.literal_eval(x))
		if sampling_frequency == 100:
			file_col = 'filename_lr'
		elif sampling_frequency == 500:
			file_col = 'filename_hr'
		
		# slice dataframe depending on strat fold for train/test split
		if subset is None:
			self.meta = df
		elif subset == 'train':
			self.meta = df[df.strat_fold <= 8]
		elif subset == 'val':
			self.meta = df[df.strat_fold == 9]
		elif subset == 'test':
			self.meta = df[df.strat_fold == 10]
		
		# raw data paths, each element=path to 1 recording
		self.paths = [os.path.join(root_path, f) for f in self.meta[file_col]]

		# create labels
		if task == 'mi_detection':
			self.labels, self.classes, _ = utils.create_mi_superclass_labels(self.meta, root_path, False)
		elif task == 'mi_norm_detection':
			self.labels, self.classes, mask = utils.create_mi_superclass_labels(self.meta, root_path, True)
			self.meta = self.meta.loc[mask]
			self.paths = self.paths[mask]

		# load preprocessor from file if needed
		self.preprocessor = None
		if mode == 'prepped':
			assert preprocessor_code is not None, 'Preprocessor code was not specified in prepped mode'
			self.preprocessor = PTBXL_Preprocessor(task=self.task)
			self.preprocessor.load(preprocessor_code)
			# encode labels immediately
			self.labels = self.preprocessor.binarize_labels(self.labels, multi=(len(self.classes) != 2))

		# sanity check
		assert(len(self.labels) == len(self.paths))

		# add padding or cropping preprocessing
		self.resize_len = resize_len if resize_len else None

	def __len__(self):
		return len(self.paths)
	
	def __getitem__(self, index):
		path = self.paths[index]
		x, _ = wfdb.rdsamp(path)
		x = np.array(x)
		x = tensor(x, dtype=torch.float32).permute(1,0)

		if self.resize_len is not None:
			x = resize_ecg(x, self.resize_len)
		
		y = self.labels[index]

		if self.debug:
			return x, y, index
		else:
			return x, y

def create_PTBXL_dataloader(sampling_frequency, preprocessor, task='superdiagnostic', subset=None, batch_size=128, resize_len=None, debug=False):
	dataset = PTBXL(sampling_frequency=sampling_frequency, subset=subset, task=task, resize_len=resize_len, mode='prepped', preprocessor_code=preprocessor, debug=debug)
	return DataLoader(dataset, batch_size=batch_size, shuffle=(not debug), num_workers=4)

from sklearn.preprocessing import StandardScaler, LabelBinarizer, MultiLabelBinarizer

class PTBXL_Preprocessor():
	def __init__(self, task='mi_detection'):
		'''Initialize PTB-XL preprocessor. All parameters are fixed.'''

		# fitted dataset
		print("Initializing fitting dataset")
		self.dataset = PTBXL(sampling_frequency=500, subset='train', task=task, mode='raw')
		self.batch_size = 128
		self.loader = DataLoader(self.dataset, batch_size=self.batch_size, num_workers=4)

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
		mean_tnsr = torch.tensor(self.ch_means, device=device, dtype=torch.float32)[None, :, None]
		std_tnsr = torch.tensor(self.ch_stds, device=device, dtype=torch.float32)[None, :, None]
		
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
			return torch.tensor(self.multi_binarizer.transform(labels), dtype=torch.float32)
		else:
			labels = np.array([x[0] for x in labels])
			return torch.tensor(self.bin_binarizer.transform(labels), dtype=torch.float32)
		
	def save(self, file_name, folder_path='../myoutput/preprocessor/'):
		'''`.pkl` extension added automatically'''
		helper.check_create_folder(folder_path)
		with open(os.path.join(folder_path, file_name+'.pkl'), 'wb') as f:
			pickle.dump(self.__dict__, f)
	
	def load(self, file_name, folder_path='../myoutput/preprocessor/'):
		'''`.pkl` extension added automatically'''
		p = os.path.join(folder_path, file_name+'.pkl')
		if not os.path.exists(p):
			raise FileNotFoundError(f"The preprocessor was not found at {p}")
		else:
			with open(p, 'rb') as f:
				attrs = pickle.load(f)
			self.__dict__.update(attrs)