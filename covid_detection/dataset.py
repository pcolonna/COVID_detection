import os
import random
import shutil

import config
import torch
from PIL import Image


class ChestXRayDataset(torch.utils.data.Dataset):
    def __init__(self, image_dirs, transform):
        # transform will do basic image augmentation
        # we will do no image augmentation on test set but we will still
        # convert to tensor and normalize

        prepare_train_test_set(image_dirs)

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


def prepare_train_test_set(image_dirs):
    class_names = ["normal", "viral", "covid"]
    root_data_dir = config.root_data_dir

    if os.path.isdir(root_data_dir):

        if not os.path.isdir(os.path.join(root_data_dir, "train")):
            os.mkdir(os.path.join(root_data_dir, "train"))
            for c in class_names:
                os.mkdir(os.path.join(root_data_dir, "train", c))

        if not os.path.isdir(os.path.join(root_data_dir, "test")):
            os.mkdir(os.path.join(root_data_dir, "test"))
            for c in class_names:
                os.mkdir(os.path.join(root_data_dir, "test", c))

        # for i, d in enumerate(config.train_dirs):
        #     os.rename(os.path.join(root_dir, d), os.path.join(root_dir, class_names[i]))

        for c in class_names:
            images = [x for x in os.listdir(config.original_data_dir[c]) if x.lower().endswith("png")]
            random.shuffle(images)

            selected_test_images = images[:30]  # random.sample(images, 30)
            selected_train_images = images[30:]  # random.sample(images, 30)

            for image in selected_test_images:
                source_path = os.path.join(config.original_data_dir[c], image)
                target_path = os.path.join(root_data_dir, "test", c, image)
                shutil.copy(source_path, target_path)

            for image in selected_train_images:
                source_path = os.path.join(config.original_data_dir[c], image)
                target_path = os.path.join(root_data_dir, "train", c, image)
                shutil.copy(source_path, target_path)
