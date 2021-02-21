import torch
from PIL import Image
import os

class ChestXRayDataset(torch.utils.data.Dataset):
    def __init__(self, image_dirs, transform):
        # transform will do basic image augmentation
        # we will do no image augmentation on test set but we will still
        # convert to tensor and normalize

        def get_images(class_name):
            # enumerate the list of images
            images = [x for x in os.listdir(image_dirs[class_name]) if x.endswith("png")]
            print(f"Found {len(images)} {class_name} examples")

            return images

        self.images = {}
        self.class_names = ["normal", "viral", "covid"]

        for c in self.class_names:
            self.images[c] = get_images(c)  # we list in this dict every images per class

        self.image_dirs = image_dirs
        self.transform = transform

    def __len__(self):
        # Will return the length of the dataset.
        # Will return the total number of images in all 3 classes combined
        return sum([len(self.images[c]) for c in self.class_names])

    def __getitem__(self, index):
        # There are way more normal and pneumonia than covid cases.
        # We want to avoid any class imbalance that could cause trouble

        class_name = random.choice(self.class_names)
        index = index % len(self.images[class_name])  # this avoid any out of bound index

        # now we select the image
        image_name = self.images[class_name][index]

        image_path = os.path.join(self.image_dirs[class_name], image_name)

        # We use Pillow to open the image
        # We convert to RGB to be compliant with ResNet.
        # We will use its pre-trained weights
        # If i were to do it from scratch i could use a channel value of 1
        image = Image.open(image_path).convert("RGB")

        # PyTorch doesn't understand Pillow images.
        # self.transform will transform it to a tensor and normalize
        return self.transform(image), self.class_names.index(class_name)
