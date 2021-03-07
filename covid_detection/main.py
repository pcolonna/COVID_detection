import logging
import os

import config
import dataset
import torch
import train
import transform

logging.basicConfig(format="%(asctime)s     %(levelname)s   %(message)s", level=logging.INFO)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device} \n")


def main():

    batch_size = 16

    train_dirs = config.train_dirs
    test_dirs = config.test_dirs

    train_transform, test_transform = transform.get_transforms()
    train_dataset = dataset.ChestXRayDataset(train_dirs, train_transform)
    test_dataset = dataset.ChestXRayDataset(test_dirs, test_transform)

    dl_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    dl_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    logging.info(f"Num of training batches {len(dl_train)}")
    logging.info(f"Num of test batches {len(dl_test)}\n")

    train.main(dl_train, dl_test, test_dataset, epochs=10)


if __name__ == "__main__":
    main()
