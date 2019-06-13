import numpy as np
import os
from os.path import join
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch 
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import transforms
from os import listdir
import pandas as pd

class ToRGB(object):

	def __init__(self):
		pass
		
	def __call__(self, sample):

		sample = sample.convert('RGB')
		return sample


cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
print(device)
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])

transform = transforms.Compose([
	ToRGB(),
	transforms.Resize((32, 32)),
	transforms.ToTensor(),
	transforms.Normalize(mean, std)
	])


class DATASET(Dataset):

	def __init__(self, img_path, label_path, transforms=None):
		super(DATASET, self).__init__()

		self.img_path = join(os.getcwd(), 'dataset', img_path, label_path)
		self.label_path = join(os.getcwd(), 'dataset', img_path, label_path+'.csv')
		
		self.label_numpy = pd.read_csv(self.label_path).values[:, 1]
		self.imgs_files = listdir(self.img_path)
		
		self.transform = transform
	
	def __len__(self):
		return len(self.imgs_files)


	def __getitem__(self, idx):
		
		filename = self.imgs_files[idx]
		img = self.transform(Image.open(join(self.img_path, filename)))
		
		label = self.label_numpy[idx]
		label = torch.LongTensor([label])

		return img, label


	