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

mean, std = np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])


class DATASET(Dataset):

	def __init__(self, img_path, label_path, transforms):
		super(DATASET, self).__init__()

		self.img_path = join(os.getcwd(), img_path)
		self.label_path = join(os.getcwd(), label_path)
		self.label_numpy = pd.read_csv(self.label_path).values[:, 1]
		self.imgs_files = listdir(self.img_path)
		self.transform = transforms
	
	def __len__(self):
		return len(self.imgs_files)


	def __getitem__(self, idx):
		
		filename = self.imgs_files[idx]
		img = self.transform(Image.open(join(self.img_path, filename)))
		
		label = self.label_numpy[idx]
		label = torch.LongTensor([label])

		return img, label


	