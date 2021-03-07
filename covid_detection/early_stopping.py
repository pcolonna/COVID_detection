import numpy as np
import torch

class EarlyStopping:

	def __init__(self, patience):
		self.patience = patience
		self.counter = 0
		self.best_val_loss = None
		self.early_stop = False

	def __call__(self, val_loss, model):

		if self.best_val_loss == None:
			self.best_val_loss = val_loss

		elif val_loss > self.best_val_loss:
			self.counter += 1
			if self.counter >= self.patience:
				self.early_stop = True

		else:
			self.best_val_loss = val_loss
			self.counter = 0