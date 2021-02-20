from matplotlib import pyplot as plt


def show_images(images, labels, preds):
    """Display six images from the dataset"""

    plt.figure(figsize=(8, 4))

    for i, image in enumerate(images):
        plt.subplot(1, 6, i + 1, xticks=[], yticks=[])

        # We need to convert the tensor image to a numpy array
        # and then transpose cause it's channel first in ResNet
        image = image.numpy().transpose((1, 2, 0))

        # We also need to undo the mean and std transform
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        image = image * std + mean

        # We clip the values between zero and one
        image = np.clip(image, 0.0, 1.0)

        plt.imshow(image)

        # We will use the x label to show the ground thruth
        # and the y label to show the prediction
        col = "green" if preds[i] == labels[i] else "red"

        plt.xlabel(f"{class_names[int(labels[i].numpy())]}")
        plt.ylabel(f"{class_names[int(preds[i].numpy())]}", color=col)

    plt.tight_layout()
    plt.show()


def show_preds(resnet18):
    """We define a method to show prediction, using the previous method to show images"""
    # We set the model to evaluation mode
    resnet18.eval()

    # We iterate over the test dataloader
    images, labels = next(iter(dl_test))

    # and get the output
    outputs = resnet18(images)

    # once we get the output, we look for the index of the maximum value
    _, preds = torch.max(outputs, 1)  # we look at dimension 1 the ouptu values

    show_images(images, labels, preds)
