import torch
import dataset
import transform
import train

def main():
    batch_size = 6


    train_dirs = {
        'normal': 'data/COVID-19 Radiography Database/normal',
        'viral': 'data/COVID-19 Radiography Database/viral',
        'covid': 'data/COVID-19 Radiography Database/covid'
    }


    test_dirs = {
        'normal': 'data/COVID-19 Radiography Database/test/normal',
        'viral': 'data/COVID-19 Radiography Database/test/viral',
        'covid': 'data/COVID-19 Radiography Database/test/covid'
    }

    train_transform, test_transform = transform.get_transforms()
    train_dataset = dataset.ChestXRayDataset(train_dirs, train_transform)
    test_dataset = dataset.ChestXRayDataset(test_dirs, test_transform)

    dl_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    dl_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print("Num of training batches", len(dl_train))
    print("Num of test batches", len(dl_test))

if __name__ == '__main__':
    main()