
# We can create a transformation object and pass a list of transformation that you want.

# We have to perform a few steps
#     - Since we are using ResNet we have to convert the images to 
#       the size the model expect
#     - apply random horizontal flip for data augmentation
#     - normalize the data in the same way as the imagenet data was normalized

import torchvision

def get_transforms():
	train_transform = torchvision.transforms.Compose([
	    torchvision.transforms.Resize(size=(224, 224)),
	    torchvision.transforms.RandomHorizontalFlip(),
	    torchvision.transforms.ToTensor(),
	    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
	                                     std=[0.229, 0.224, 0.225])
	])

	# We don't want any data augmentation on the test set
	test_transform = torchvision.transforms.Compose([
	    torchvision.transforms.Resize(size=(224, 224)),
	    torchvision.transforms.ToTensor(),
	    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
	                                     std=[0.229, 0.224, 0.225])
	])

	return train_transform, test_transform