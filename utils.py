import os
import glob
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wfdb
import ast
from sklearn.metrics import fbeta_score, roc_auc_score, roc_curve, roc_curve, auc
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer


#=====================DATA PROCESSING========================

def load_dataset(path, sampling_frequency):
	"""
	Load dataset from csv. Raw data is read from file path stored in csv.

	Args:
		path (str): Folder where the data (`*_database.csv`) is stored
		sampling_frequency (int): 100 if using 100 Hz (low res) data, or 500 if using 500 Hz (high res) data.

	Returns:
		tuple:
			np.ndarray: Raw ECG signal data.
			pd.DataFrame: Corresponding metadata including labels.
	"""
	print("Loading dataset")

	if path.split('/')[-2] == 'ptbxl':
		# load data from data folder and convert annotation data (scp_codes column from string to dict)
		Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id', na_values=[""])
		Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

		# Load raw signal data
		X = load_raw_data_ptbxl(Y, sampling_frequency, path)

	return X, Y

def load_raw_data_ptbxl(df, sampling_rate, path):
	bin_data_path = ""
	if sampling_rate == 100:
		bin_data_path = path+'raw100.npy'
	else:
		bin_data_path = path+'raw500.npy'

	if os.path.exists(bin_data_path):
		data = np.load(bin_data_path, allow_pickle=True)
	else:
		data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_lr)]
		data = np.array([signal for signal, meta in data])
		pickle.dump(data, open(bin_data_path, 'wb'), protocol=4)

	print("Dataset loaded")
	return data

def compute_label_aggregations(df, folder, ctype):
	"""
	Maps SCP codes in the dataset to task-related labels using `scp_statements.csv` table, and adds it to df

	Args:
		df (pd.DataFrame): The dataset to process, containing SCP codes.
		folder (str): Directory containing the `scp_statements.csv` file, which defines SCP codes and their related labels.
		ctype (str): The classification task type, which determines the labels to be used.

	Returns:
		pd.DataFrame: The dataset with a new column containing the task-related labels.
	"""

	# Read SCP code descriptors from its own table
	# 'diagnostic', 'form', and 'rhythm' are multi-hot columns containing either 1 or 0
	aggregation_df = pd.read_csv(folder+'scp_statements.csv', index_col=0)

	def aggregate_labels(y_dic, lookup_df, label_col=None):
		'''
		Extracts a unique list of labels from `lookup_df` based on input dict.
		Dict keys are assumed to be in the df's `index_col`.
		If `label_col` is not given, uses `index_col` for the labels.
		'''
		tmp = []
		for key in y_dic.keys():
			if key in lookup_df.index:
				if label_col is None:
					tmp.append(key)
				else:
					label = lookup_df.loc[key, label_col]
					if pd.notna(label):
						tmp.append(label)
		return list(set(tmp))
	
	col = ctype if ctype != 'all' else 'all_scp'

	if ctype in ['diagnostic', 'subdiagnostic', 'superdiagnostic']:
		diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]
		label_column = {
			'diagnostic': None,
			'subdiagnostic': 'diagnostic_subclass',
			'superdiagnostic': 'diagnostic_class'
		}[ctype]
		df[col] = df.scp_codes.apply(lambda x: aggregate_labels(x, diag_agg_df, label_column))

	elif ctype in ['form', 'rhythm']:
		filt_agg_df = aggregation_df[aggregation_df[ctype] == 1.0]
		df[col] = df.scp_codes.apply(lambda x: aggregate_labels(x, filt_agg_df))

	elif ctype == 'all':
		df[col] = df.scp_codes.apply(lambda x: list(set(x.keys())))

	return df

def select_data(X_raw, Y_raw, ctype, min_samples, outputfolder):
	'''
	Selects data from input by minimum samples filtering.
	Creates a n * c multi-hot class array using MultiLabelBinarizer.

	Args:
		X_raw (np.ndarray): Raw signal data loaded using `load_dataset`
		Y_raw (pd.DataFrame): Signal metadata with class labels.
		ctype (str): Type of SCP code classification
		min_samples (int): Minimum samples required for a class to be included.
		outputfolder (str): Path to save the MultiLabelBinarizer.

	Returns:
		tuple:
			np.ndarray: Signal data selected using min_samples
			pd.DataFrame: Metadata selected using min_samples
			np.ndarray: Multi-hot labels (does not include label names)
			MultiLabelBinarizer: The binarizer used for multi-hot data creation
	'''
	mlb = MultiLabelBinarizer()
	labels_col = ctype if ctype != 'all' else 'all_scp'

	if ctype in ['subdiagnostic', 'superdiagnostic', 'form', 'rhythm', 'all']:
		counts = pd.Series(np.concatenate(Y_raw[labels_col].values)).value_counts()
		counts = counts[counts > min_samples]
		Y_raw[labels_col] = Y_raw[labels_col].apply(lambda x: list(set(x).intersection(set(counts.index.values))))
	Y_raw[labels_col+'_len'] = Y_raw[labels_col].apply(len)

	X = X_raw[Y_raw[labels_col+'_len'] > 0]
	Y = Y_raw[Y_raw[labels_col+'_len'] > 0]
	mlb.fit(Y[labels_col].values)
	y = mlb.transform(Y[labels_col].values)

	print(mlb.classes_)

	# save LabelBinarizer
	with open(outputfolder+'mlb.pkl', 'wb') as tokenizer:
		pickle.dump(mlb, tokenizer)

	return X, Y, y, mlb

def preprocess_signals(X_train, X_validation, X_test, outputfolder):
	# Standardize data such that mean 0 and variance 1
	ss = StandardScaler()
	ss.fit(np.vstack(X_train).flatten()[:,np.newaxis].astype(float))
	
	# Save Standardizer data
	with open(outputfolder+'standard_scaler.pkl', 'wb') as ss_file:
		pickle.dump(ss, ss_file)

	return apply_standardizer(X_train, ss), apply_standardizer(X_validation, ss), apply_standardizer(X_test, ss)

def apply_standardizer(X, ss):
	X_tmp = []
	for x in X:
		x_shape = x.shape
		X_tmp.append(ss.transform(x.flatten()[:,np.newaxis]).reshape(x_shape))
	X_tmp = np.array(X_tmp)
	return X_tmp

#=====================EVALUATION========================

def get_appropriate_bootstrap_samples(y_true, n_bootstrapping_samples):
	"""
	Generates bootstrap samples by randomly sampling indices from y_true with replacement.
	Each sample is guaranteed to have all classes occur at least once.

	Args:
		y_true (np.ndarray): Multi-hot true labels array being sampled, of shape (N, K)
		n_bootstrapping_samples (int): Number of bootstrap samples to generate.

	Returns:
		np.ndarray: Array of shape (n_bootstrapping_samples, N), 
		each row contains indices for y_true.
	"""
	samples=[]
	while True:
		ridxs = np.random.randint(0, len(y_true), len(y_true))
		if y_true[ridxs].sum(axis=0).min() != 0:
			samples.append(ridxs)
			if len(samples) == n_bootstrapping_samples:
				break
	return np.stack(samples)

def generate_results(idxs, y_true, y_pred):
	return evaluate_experiment(y_true[idxs], y_pred[idxs])

def evaluate_experiment(y_true, y_pred):
	"""
	Evaluates the ROC AUC of y_pred against y_true
	
	Args:
		y_true (np.ndarray): 2D array size (N, K) with values 1 or 0
		y_pred (np.ndarray): 2D array size (N, K) with values between 0 and 1

	Returns:
		pd.DataFrame: the evaluation result, containing only macro AUC score
	"""
	results = {}
	results['macro_auc'] = roc_auc_score(y_true, y_pred, average='macro')
	df_result = pd.DataFrame(results, index=[0])
	return df_result