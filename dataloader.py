import pandas as pd
import numpy as np
import wfdb
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import ast
import utils
from ecg_processing import resize_ecg_array
import helper
from tqdm import tqdm
import pickle
import os
import warnings
from collections import Counter
from itertools import chain

default_task = 'superdiagnostic'

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
	def __init__(self, root_path='../data/ptbxl/', sampling_frequency:int=100, subset=None, task=default_task, multilabel=True, task_use_other=False, resize_len=None, is_raw=False, data_root_path='../mydata/', debug=False):
		'''
		Initialize the dataset.

		The dataset slice to load depends on the data subset and the task (multilabel or MI classification)

		**Note**: preprocessed signals should be stored as individual signals in .npy files, with the same folder structure as the original data.

		Args:
			root_path (str): The path containing the dataset (ptbxl_database.csv, records folders, and SCP descriptor scp_statements.csv)
			sampling_frequency (int): The sampling frequency being used (100/500 Hz)
			subset (str or None): The subset of data being loaded ('train', 'val', 'test', or None for all)
			task (str): The classification task, should be either: superdiagnostic, mi_detection or mi_norm_detection
			multilabel (bool): If True, use MultiLabelBinarizer for label mapping, otherwise use LabelBinarizer.
			task_use_other (bool): If True, include classes outside of the task scope as "Other" class.
			resize_len (int or None): Length in timesteps to resize the records into
			is_raw (bool): If True, uses raw signals, otherwise uses preprocessed signals.
			data_root_path (str): Preprocessed signals root folder (if using preprocessed signals).
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
		self.multilabel = multilabel
		self.task_use_other = task_use_other
		self.resize_len = resize_len
		self.is_raw = is_raw
		self.data_root_path = data_root_path
		self.debug = debug

		self.shape = (12, sampling_frequency*100)

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
		if self.is_raw:
			self.paths = [os.path.join(root_path, f) for f in self.meta[file_col]]
		else:
			self.paths = [os.path.join(data_root_path, f+'.npy') for f in self.meta[file_col]]
		self.paths = np.array(self.paths)

		# compute labels using preprocessor
		labelprep = LabelPreprocessor()
		binarizer_code = self.task + ('_other' if self.task_use_other else '')
		labels, self.exist_classes, mask = labelprep.compute_labels(self.meta, self.task, self.task_use_other)

		# filter paths & metadata based on mask
		self.paths = self.paths[mask]
		self.meta = self.meta[mask]

		# binarize labels
		if self.multilabel:
			self.labels, self.classes = labelprep.multi_binarize(labels, binarizer_code)
		else:
			self.labels, self.classes = labelprep.binarize(labels, binarizer_code)

		if len(self.classes) > len(self.exist_classes):
			diff_classes = []
			for c in self.classes:
				if c not in self.exist_classes:
					diff_classes.append(c)
			warnings.warn(
				f"The number of classes in {self.subset} subset ({len(self.exist_classes)}) is different from "
				f"the number of classes in the entire dataset ({len(self.classes)}). "
				f"Missing: {diff_classes}",
				UserWarning,
				stacklevel=2)

		# sanity check
		assert(self.labels.shape[0] == self.paths.shape[0]), f'{self.labels.shape[0]=} not equal to {self.paths.shape[0]=}'

	def __len__(self):
		return len(self.paths)
	
	def __getitem__(self, index):
		path = self.paths[index]
		# load signal from file
		if self.is_raw:
			x, _ = wfdb.rdsamp(path)
			# x will have dimension (N, C) so we apply transpose
			x = np.array(x, dtype=np.float32).T
		else:
			x = np.load(path)

		# load labels from attributes
		y = self.labels[index].astype(np.float32)

		# shrink or extend ECG length if needed
		if self.resize_len is not None:
			x = resize_ecg_array(x, self.resize_len)
		
		if self.debug:
			return x, y, index

		return x, y

def create_PTBXL_dataloader(sampling_frequency, task='superdiagnostic', subset=None, batch_size=128, resize_len=None, debug=False):
	task_config = {
		'superdiagnostic': {
			'multilabel':True,
			'task_use_other':False
		},
		'mi_detection': {
			'multilabel':False,
			'task_use_other':True
		},
		'mi_norm_detection': {
			'multilabel':False,
			'task_use_other':False
		}
	}
	config = task_config[task]
	multilabel = config['multilabel']
	task_use_other = config['task_use_other']
	dataset = PTBXL(
		sampling_frequency=sampling_frequency,
		subset=subset,
		task=task,
		multilabel=multilabel,
		task_use_other=task_use_other,
		resize_len=resize_len,
		debug=debug
	)
	return DataLoader(dataset, batch_size=batch_size, shuffle=(not debug), num_workers=4, pin_memory=True)

from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from scipy.signal import firwin

def create_fir_tensor(n_taps, f, fs, ch=12):
	return torch.tensor(firwin(n_taps, f, fs=fs), dtype=torch.float32).view(1,1,-1).repeat(ch,1,1)

class LabelPreprocessor():
	def __init__(self, scp_path='../data/ptbxl/'):
		'''Initialize the label preprocessor.
		
		Args:
			scp_path (str, optional): Path to SCP descriptor file. Default is `"../data/ptbxl/"`'''
		self.scp_descriptor = pd.read_csv(scp_path+'scp_statements.csv', index_col=0)

	def aggregate_sample_labels(self, y_dic, scp_desc, label_col=None, target_classes=None):
		'''
		Maps SCP codes of a single recording into labels based on task criteria.

		Args:
			y_dic (dict): Dictionary containing raw SCP codes of a single recording. Key=SCP code, value=confidence level.
			scp_desc (pd.DataFrame): Dataframe of SCP codes with description (most importantly, the type (diagnostic/form/rhythm) and diagnostic classes & subclasses).
			label_col (str or None): If set, replaces the dict SCP code with the value of this column in `scp_desc`.
			target_classes (list of str or None): If set, only the labels listed here will be added to the new label.

		Returns:
			list: a list of strings.
		'''
		result = set()
		for code in y_dic.keys():
			if code in scp_desc.index:
				if label_col is None:
					result.add(code)
				else:
					# get label from scp descriptor column, then append if in target_classes
					# always append if target_classes is unspecified
					label = scp_desc.loc[code, label_col]
					if pd.notna(label) and (target_classes is None or label in target_classes):
						result.add(label)
		return list(result)

	def compute_labels(self, df, task, use_other=False):
		'''
		Maps SCP codes of an entire dataset into classification labels based on classification task.

		Args:
			df (pd.DataFrame): The ECG recording metadata dataset which lists the SCP codes for each recording.
			task (str): The classification task at hand.
			use_other (bool): If True, recordings that don't match any label criteria will be considered for classification as an "Other" class. Otherwise, they will be discarded.
		
		Returns:
			tuple:
			- labels (np.ndarray): An array of lists of labels. May be shorter than the original DF if `use_other=False`.
			- classes (list): A list of all classes present in `labels`.
			- mask (np.ndarray): An array of boolean with the same length as `df`. True values indicate the `df` row is included in `labels`, excluded otherwise.
		'''

		# setup config based on task
		# TODO: Add other configs here if needed
		task_config = {
			'mi_detection': {'code_type': 'diagnostic', 'label_col': 'diagnostic_class', 'targets': ['MI']},
			'mi_norm_detection': {'code_type': 'diagnostic', 'label_col': 'diagnostic_class', 'targets': ['MI', 'NORM']},
			'diagnostic': {'code_type': 'diagnostic', 'label_col': None, 'targets': None},
			'superdiagnostic': {'code_type': 'diagnostic', 'label_col': 'diagnostic_class', 'targets': None},
			'subdiagnostic': {'code_type': 'diagnostic', 'label_col': 'diagnostic_subclass', 'targets': None}
		}
		config = task_config.get(task)
		if config is None:
			raise ValueError(f"Unknown task: {task}. Valid tasks: {list(task_config.keys())}")
		if not use_other and (config['targets'] is not None and len(config['targets']) == 1):
			warnings.warn(
				f"{task=} has single target class {config['targets']} but use_other=False. "
				"This will cause non-matching samples to have empty labels. "
				"Recommended to set use_other=True for binary classification tasks.",
				UserWarning,
				stacklevel=2)

		scp_desc = self.scp_descriptor[self.scp_descriptor[config['code_type']] == 1.0]

		classes = set()
		def process_labels(y_dic):
			labels = self.aggregate_sample_labels(
				y_dic, 
				scp_desc=scp_desc,
				label_col=config['label_col'],
				target_classes=config['targets']
			)
			if not labels and use_other:
				labels = ['OTHER']
			classes.update(labels)
			return labels

		labels = df.scp_codes.apply(lambda x: process_labels(x))
		# labels is a pd.Series of lists
		classes = list(classes)
		mask = labels.apply(bool)
		labels = labels[mask]
		labels = labels.to_numpy()
		mask = mask.to_numpy()
		return labels, classes, mask
	
	def get_class_counts(self, labels):
		'''Args:
			labels (np.ndarray): Array of lists containing labels.
		
		Returns:
			Counter: a dict-like object containing the count of each class.'''
		return Counter(chain.from_iterable(labels))

	def binarize_fit_save(self, classes, codename, folder_path='../myoutput/binarizers/'):
		'''
		Fits a LabelBinarizer and MultiLabelBinarizer for label transformation, then saves them to a file.

		Args:
			classes (list of str): List containing all class labels.
			codename (str): A name describing the binarizer task.
			folder_path (str, optional): The path to save the binarizers to. Default is `'../myoutput/binarizers'`.

		Returns:
			classes (list of str): List of classes now ordered by their transformation values (alphabetical order).
		'''
		# label binarizer
		lb = None
		if (len(classes) == 2):
			lb = LabelBinarizer()
			lb.fit(classes)
		
		# multilabel binarizer
		mlb = MultiLabelBinarizer()
		wrapped_classes = [[c] for c in classes]
		mlb.fit(wrapped_classes)

		helper.check_create_folder(folder_path)
		self.save_binarizer(mlb, codename+'_mlb', folder_path)
		if lb is not None:
			self.save_binarizer(lb, codename+'_lb', folder_path)

		return mlb.classes_
	
	def save_binarizer(self, b, codename, folder_path='../myoutput/binarizers/'):

		helper.check_create_folder(folder_path)
		with open(os.path.join(folder_path, codename+'.pkl'), 'wb') as f:
			pickle.dump(b, f)
	
	def load_binarizer(self, codename, folder_path='../myoutput/binarizers/'):
		with open(os.path.join(folder_path, codename+'.pkl'), 'rb') as f:
			return pickle.load(f)
	
	def binarize(self, labels, codename, folder_path='../myoutput/binarizers/', empty_default=0):
		lb = self.load_binarizer(codename+'_lb', folder_path)
		tmp = []
		for l in labels:
			tmp.append(l[0] if l else lb.classes_[empty_default])
		labels = tmp
		return lb.transform(labels), lb.classes_
	
	def multi_binarize(self, labels, codename, folder_path='../myoutput/binarizers/'):
		mlb = self.load_binarizer(codename+'_mlb', folder_path)
		return mlb.transform(labels), mlb.classes_

from scipy.signal import butter, filtfilt

class SignalPreprocessor():
	def __init__(self):
		'''Initialize PTB-XL preprocessor. All parameters are fixed.
		All preprocessing is done on CPU with numpy arrays (not torch tensors),
		except for fitting (for now).'''

		# general attributes
		self.n_channels = 12
		self.subset = None
		self.dataset = None
		self.loader = None
		self.fs = None

		# computed attributes
		self.ch_means = None
		self.ch_stds = None
		self.ch_means_arr = None
		self.ch_stds_arr = None

	def build_loader(self, fs=500, subset='train'):
		# fitted dataset
		print("Initializing fitting dataset")
		self.fs = fs
		self.subset = subset
		self.batch_size = 128
		self.dataset = PTBXL(sampling_frequency=fs, subset=self.subset, is_raw=True, task_use_other=True, debug=True)
		self.loader = DataLoader(self.dataset, batch_size=self.batch_size, num_workers=4)

	def fit_scaler(self):
		# Finds mean and std from the training set
		if self.loader is None or self.fs != 500:
			self.build_loader(fs=500)
		
		ch_sums = np.zeros(self.n_channels, dtype=np.float32)
		ch_sums_sq = np.zeros(self.n_channels, dtype=np.float32)
		n_total = 0

		for ecgs, _, _ in tqdm(self.loader, desc='Computing scaler parameters'):
			ecgs = ecgs.numpy()
			ecgs = self.denoise(ecgs, fs=self.fs)
			ch_sums += ecgs.sum(axis=(0,2))
			ch_sums_sq += (ecgs ** 2).sum(axis=(0,2))
			n_total += ecgs.shape[0] * ecgs.shape[2]

		mean = ch_sums/n_total
		std = np.sqrt((ch_sums_sq / n_total) - (mean ** 2))
		self.ch_means = mean
		self.ch_stds = std
		self.ch_means_arr = np.array(self.ch_means, dtype=np.float32)[None, :, None]
		self.ch_stds_arr = np.array(self.ch_stds, dtype=np.float32)[None, :, None]
	
	def normalize(self, ecgs):
		# expects an np.array
		if ecgs.dtype != np.float32:
			ecgs = ecgs.astype(np.float32)
		single = False
		if ecgs.ndim == 2:
			single = True
			ecgs = np.expand_dims(ecgs, axis=0)
		assert ecgs.shape[1] == self.n_channels, f'Expected {self.n_channels} leads, got {ecgs.shape[1]}'

		ret = (ecgs - self.ch_means_arr) / self.ch_stds_arr
		if single:
			ret = ret.squeeze(0)
		return ret
	
	def butter_lowpass(self, fs, cutoff, order=4):
		nyq = 0.5 * fs
		norm_cutoff = cutoff / nyq
		b, a = butter(order, norm_cutoff, btype='low', analog=False)
		return b, a

	def apply_lowpass(self, ecgs, fs, cutoff, order=4, pad_len=None):
		'''ecgs should be an NP array, with shape (B, C, N) or (C, N).'''
		assert ecgs.shape[-2] == self.n_channels, f"Expected {self.n_channels} on axis -2, got {ecgs.shape[-2]}"
		single = False
		if ecgs.ndim == 2:
			single = True
			ecgs = np.expand_dims(ecgs, axis=0)

		# pad signal to prevent transient artifact
		if pad_len is None:
			pad_len = fs
		start_reflect = ecgs[:, :, :pad_len][:, :, ::-1]
		end_reflect = ecgs[:, :, -pad_len:][:, :, ::-1]
		padded = np.concatenate([start_reflect, ecgs, end_reflect], axis=-1)

		b, a = self.butter_lowpass(fs, cutoff, order)
		filtered = filtfilt(b, a, padded, axis=2)
		if single:
			filtered = filtered.squeeze(0)
			return filtered[:, pad_len:-pad_len]
		return filtered[:, :, pad_len:-pad_len]
	
	def denoise(self, ecgs, fs, cutoff_lo=0.5, cutoff_hi=40):
		'''
		Applies the following transformations to the ECGs in order:
		- Baseline removal with cutoff frequency=0.5 Hz
		- Zeroing the mean
		- High frequency noise removal with default cutoff frequency=40 Hz
		
		ecgs should have the last dim be the time dimension.
		'''
		baseline = self.apply_lowpass(ecgs, fs, cutoff_lo)
		ret = ecgs - baseline
		ret = ret - ret.mean(axis=-1, keepdims=True)
		ret = self.apply_lowpass(ret, fs, cutoff_hi)
		return ret
	
	def preprocess_all(self, fs, root_path='../mydata/', dry=False):
		assert fs in [100, 500], 'fs must be either 100 or 500'

		if self.loader is None or self.fs != fs:
			self.build_loader(fs, subset=None)

		helper.check_create_folder(root_path)
		if fs == 100:
			file_col = 'filename_lr'
		elif fs == 500:
			file_col = 'filename_hr'
		col_idx = self.dataset.meta.columns.get_loc(file_col)

		overwrite = False
		skip = False

		bar = tqdm(self.dataset, desc='Preprocessing all to file' + (' [dry]' if dry else ''), unit='record', leave=True)
		for ecg, _, i in bar:
			denoised = self.denoise(ecg, self.fs)
			normalized = self.normalize(denoised)

			path = self.dataset.meta.iloc[i, col_idx]
			filename = os.path.join(root_path, path+'.npy')
			fileloc = os.path.dirname(filename)
			helper.check_create_folder(fileloc)

			# save processed ecg to file
			if not dry:
				bar.set_postfix_str(f'Attempting write: {filename} to {fileloc}')
				if not overwrite:
					if not os.path.exists(filename):
						np.save(filename, normalized)
					else:
						if skip:
							tqdm.write(f'Skipping write at {filename}')
							continue
						inp = input('File already exists. o=overwrite|s=skip|oa=overwrite all|sa=skip all\n>')
						if inp[0] == 'o':
							np.save(filename, normalized)
							tqdm.write(f'Overwriting file at {filename}')
							if inp == 'oa':
								overwrite = True
						elif inp[0] == 's':
							tqdm.write(f'Skipping write at {filename}')
							if inp[0] == 'sa':
								skip = True
							continue
				else:
					if os.path.exists(filename):
						tqdm.write(f'Overwriting file')
					np.save(filename, normalized)
		bar.set_description('Finished preprocessing')
	
	def save(self, codename, folder_path='../myoutput/preprocessor/'):
		'''`.pkl` extension added automatically'''
		helper.check_create_folder(folder_path)
		attrs = {
			'ch_means': self.ch_means,
			'ch_stds': self.ch_stds,
			'ch_means_arr': self.ch_means_arr,
			'ch_stds_arr': self.ch_stds_arr
		}
		with open(os.path.join(folder_path, codename+'.pkl'), 'wb') as f:
			pickle.dump(attrs, f)
	
	def load(self, codename, folder_path='../myoutput/preprocessor/'):
		'''`.pkl` extension added automatically'''
		p = os.path.join(folder_path, codename+'.pkl')
		if not os.path.exists(p):
			raise FileNotFoundError(f"Preprocessor state dict not found at {p}")
		else:
			with open(p, 'rb') as f:
				attrs = pickle.load(f)
			self.__dict__.update(attrs)