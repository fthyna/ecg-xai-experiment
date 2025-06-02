import torch
from tqdm import tqdm

# TODO - move training pipeline here
class ModelTrainer():
	def __init__(self, model, num_epochs, loader_train, loader_val, device, optimizer='adam', lr=1e-3):
		self.model = model
		self.num_epochs = num_epochs
		self.loader_train = loader_train
		self.loader_val = loader_val
		self.device = device
		self.

	def train(self):
		train_losses = []
		val_losses = []

		for epoch in range(self.num_epochs):
			# training
			self.model.train()
			train_loss = 0.0

			loader_train_tqdm = tqdm(self.loader_train, desc=f'{epoch+1}/{self.num_epochs} [Train]', leave=True)

			for batch in loader_train_tqdm:
				inputs, labels = [x.to(self.device) for x in batch]

				optimizer.zero_grad()
				outputs = model(inputs)
				loss = criterion(outputs, labels)
				with torch.autograd.set_detect_anomaly(True):
					loss.backward()
				optimizer.step()

				train_loss += loss.item()

			avg_train_loss = train_loss / len(loader_train)
			train_losses.append(avg_train_loss)

			# validation
			model.eval()
			val_loss = 0.0
			loader_val_tqdm = tqdm(loader_val, desc=f'{epoch+1}/{num_epochs} [Val]  ', leave=True)
			with torch.no_grad():
				for batch in loader_val_tqdm:
					inputs, labels = [x.to(device) for x in batch]

					outputs = model(inputs)
					loss = criterion(outputs, labels)
					val_loss += loss.item()

			avg_val_loss = val_loss / len(loader_val)
			val_losses.append(avg_val_loss)

			tqdm.write(f"train loss: {avg_train_loss:.4f} | val loss: {avg_val_loss:.4f}")