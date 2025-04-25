import utils
import os
import pandas as pd
import numpy as np
import multiprocessing
from itertools import repeat
from ecg_processing import plot_ecg_record

class Experiment():
	def __init__(
			self,
			experiment_name,
			task,
			data_folder,
			output_folder,
			models,
			sampling_frequency=100,
			min_samples=0,
			train_fold=8,
			val_fold=9,
			test_fold=10,
			folds_type='strat'
			):
		print("Initializing experiment")

		self.models = models
		self.min_samples = min_samples
		self.task = task
		self.train_fold = train_fold
		self.val_fold = val_fold
		self.test_fold = test_fold
		self.folds_type = folds_type
		self.experiment_name = experiment_name
		self.output_folder = output_folder
		self.data_folder = data_folder
		self.sampling_frequency = sampling_frequency

		main_out_dir = self.output_folder + self.experiment_name
		self.experiment_folder = {
			'results': self.output_folder + self.experiment_name + '/results/',
			'models' : self.output_folder + self.experiment_name + '/models/',
			'data' : self.output_folder + self.experiment_name + '/data/'
		}
		
		if not os.path.exists(main_out_dir):
			os.makedirs(main_out_dir)
			for key, d in self.experiment_folder:
				if not os.path.exists(d):
					os.makedirs(d)
		
		print("Experiment initialized")

	def prepare(self):
		print("Preparing experiment")
		self.data, self.labels = utils.load_dataset(self.data_folder, self.sampling_frequency)

		print("Computing labels")
		self.labels = utils.compute_label_aggregations(self.labels, self.data_folder, self.task)
		self.labels_col = self.task if self.task != 'all' else 'all_scp'

		print("Selecting data")
		self.data, self.labels, self.Y, _ = utils.select_data(self.data, self.labels, self.task, self.min_samples, self.experiment_folder['data'])
		self.input_shape = self.data[0].shape
		
		print("Splitting train, val and test data")
		# choose folds for train, val and test
		self.X_test = self.data	[self.labels.strat_fold == self.test_fold]
		self.y_test = self.Y	[self.labels.strat_fold == self.test_fold]

		self.X_val = self.data	[self.labels.strat_fold == self.val_fold]
		self.y_val = self.Y		[self.labels.strat_fold == self.val_fold]

		self.X_train = self.data[self.labels.strat_fold <= self.train_fold]
		self.y_train = self.Y	[self.labels.strat_fold <= self.train_fold]

		# preprocessing
		print("Preprocessing data")
		self.X_train, self.X_val, self.X_test = utils.preprocess_signals(self.X_train, self.X_val, self.X_test, self.experiment_folder['data'])
		self.n_classes = self.y_train.shape[1]

		# save train and test labels
		print(f"Saving labels to {self.experiment_folder['data']}")
		self.y_train.dump(self.experiment_folder['data'] + 'y_train.npy')
		self.y_val.dump(self.experiment_folder['data'] + 'y_val.npy')
		self.y_test.dump(self.experiment_folder['data'] + 'y_test.npy')
		print("Experiment prepared")

	def perform(self):
		for model_description in self.models:
			modelname = model_description['modelname']
			modeltype = model_description['modeltype']
			modelparams = model_description['parameters']

			mpath = self.experiment_folder['models'] + modelname + '/'

			if not os.path.exists(mpath):
				os.makedirs(mpath)
			if not os.path.exists(mpath+'results/'):
				os.makedirs(mpath+'results/')

			if modeltype == 'fastai_model':
				from models.fastai_model import fastai_model
				model = fastai_model(modelname, self.n_classes, self.sampling_frequency, mpath, self.input_shape, **modelparams)
			
			model.fit(self.X_train, self.y_train, self.X_val, self.y_val)
			model.predict(self.X_train).dump(mpath+'y_train_pred.npy')
			model.predict(self.X_val).dump(mpath+'y_val_pred.npy')
			model.predict(self.X_test).dump(mpath+'y_test_pred.npy')
		
	def evaluate(self, n_bootstrapping_samples=100, n_jobs=20, bootstrap_eval=False, dumped_bootstraps=True):
		# get labels
		y_train = np.load(self.experiment_folder['data']+'y_train.npy')
		y_test = np.load(self.experiment_folder['data']+'y_test.npy')

		if bootstrap_eval:
			if not dumped_bootstraps:
				test_samples = np.array(utils.get_appropriate_bootstrap_samples(y_test, n_bootstrapping_samples))
			else:
				test_samples = np.load(self.output_folder+self.experiment_name+'/test_bootstrap_ids.npy', allow_pickle=True)
		else:
			test_samples = np.array([range(len(y_test))])

		test_samples.dump(self.output_folder+self.experiment_name+'/test_bootstrap_ids.npy')

		for m in sorted(os.listdir(self.experiment_folder['models'])):
			print(m)
			mpath = self.experiment_folder['models']+m+'/'
			rpath = self.experiment_folder['models']+m+'/results/'

			y_train_pred = np.load(mpath+'y_train_pred.npy', allow_pickle=True)
			y_test_pred = np.load(mpath+'y_test_pred.npy', allow_pickle=True)

			thresholds = None

			pool = multiprocessing.Pool(n_jobs)

			te_df = pd.concat(pool.starmap(utils.generate_results, zip(test_samples, repeat(y_test), repeat(y_test_pred))))
			te_df_point = utils.generate_results(range(len(y_test)), y_test, y_test_pred)
			te_df_result = pd.DataFrame(
				np.array([
					te_df_point.mean().values,
					te_df.mean().values,
					te_df.quantile(0.05).values,
					te_df.quantile(0.95).values
				]), columns=te_df.columns, index=['point', 'mean','lower','upper']
			)

			pool.close()
			te_df_result.to_csv(rpath+'te_results.csv')

	def demo(self):
		import torch

		print("Running model demo")
		for model_description in self.models:
			modelname = model_description['modelname']
			modeltype = model_description['modeltype']
			modelparams = model_description['parameters']

			print(f"Using model {modelname}")

			mpath = self.experiment_folder['models'] + modelname + '/'

			if modeltype == 'fastai_model':
				from models.fastai_model import fastai_model
				model = fastai_model(modelname, self.n_classes, self.sampling_frequency, mpath, self.input_shape, **modelparams)

			# Choose any data point
			id = 2
			y = self.Y[id]
			x = [self.data[id]]

			print(f"Showing ECG record {id} with labels {y}")
			plot_ecg_record(x[0])

			print("Predicting labels")
			y_dummy = [np.ones(5, dtype=np.float32) for _ in range(len(self.X_train))]
			learn = model._get_learner(self.X_train, y_dummy, self.X_val, y_dummy, num_classes=5)
			m = learn.model
			x_tensor = torch.tensor(x).float().unsqueeze(0)
			pred = m(x_tensor)
			print(f"Prediction result: {pred}")