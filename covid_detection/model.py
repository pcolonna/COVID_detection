import torch
import torchvision


def create():

    # First we will import the pretrained ResNet model from torchvision
    resnet18 = torchvision.models.resnet18(pretrained=True)

    # ResNet18 was trained on Imagenet, which has 1000 classes. But we only want 3.
    # so we want to change the last fully connected layer and change it to 3 output
    # features (looking at the last layer, we can see it has out_features=1000)

    # To do so, we will change resnet.fc, with fc the key of the last layer
    resnet18.fc = torch.nn.Linear(in_features=512, out_features=3)

    # Because we are doing classification, we will use an appropriate loss function,
    # e.g cross entropy loss
    loss_fn = torch.nn.CrossEntropyLoss()

    # We will use Adam as an optimizer, and we will optimize all params
    # using resent18.parameters()
    optimizer = torch.optim.Adam(resnet18.parameters(), lr=3e-5)

    return resnet18, loss_fn, optimizer
