import logging

import numpy as np
import torch
import config

logging.basicConfig(format="%(asctime)s     %(levelname)s   %(message)s", level=logging.INFO)


class EarlyStopping:
    """Early stops if the validations doesn't improve after a given patience"""

    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best_val_loss = None
        self.early_stop = False

    def __call__(self, val_loss, accuracy, model):

        if self.best_val_loss == None:
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)

        elif val_loss >= self.best_val_loss:
            self.counter += 1

            if self.counter >= self.patience:
                logging.info(f"Performance hasn't improved in {self.patience} epochs. Stopping early...")
                self.early_stop = True

        elif accuracy == 1:
            logging.info("Accuracy is 1. Stopping early...")
            self.early_stop = True

        else:
            self.best_val_loss = val_loss
            self.counter = 0
            self.save_checkpoint(val_loss, model)

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""

        logging.info(f"Validation loss decreased ({self.best_val_loss:.6f} --> {val_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), config.best_model_path)
