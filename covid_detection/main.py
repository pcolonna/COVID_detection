import os

import config
import dataset
import torch
import train
import transform


def main():
    batch_size = 6

    train_dirs = config.train_dirs
    test_dirs = config.test_dirs

    train_transform, test_transform = transform.get_transforms()
    train_dataset = dataset.ChestXRayDataset(train_dirs, train_transform)
    test_dataset = dataset.ChestXRayDataset(test_dirs, test_transform)

    dl_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    dl_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print("Num of training batches", len(dl_train))
    print("Num of test batches", len(dl_test))


if __name__ == "__main__":
    main()
