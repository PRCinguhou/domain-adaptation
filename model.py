import torch.nn as nn
import torch
import numpy as np
import grad_rever_function

import torchvision.models as models
class encoder(nn.Module):

	def __init__(self):
		super(encoder, self).__init__()

		self.cnn = nn.Sequential(
			nn.Conv2d(3, 16, 5, 1, 2),
			nn.BatchNorm2d(16),
			nn.ReLU(True),
			nn.MaxPool2d(2),

			nn.Conv2d(16, 32, 5, 1, 2),
			nn.BatchNorm2d(32),
			nn.ReLU(True),
			nn.MaxPool2d(2),
			
			nn.Dropout2d(),
			nn.Conv2d(32, 64, 5, 1, 2),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			nn.Conv2d(64, 128, 5, 1, 2),
			nn.BatchNorm2d(128),
			nn.ReLU(True)
			)

		self.fc = nn.Sequential(
			nn.Linear(7*7*128, 2048),
			nn.BatchNorm1d(2048),
			nn.ReLU(True)
			)

		self.cls = nn.Linear(2048, 10)


	def forward(self, x):

		feature = self.cnn(x)
		feature = feature.view(x.size(0), -1)
		feature = self.fc(feature)
		pred = self.cls(feature)

		return pred, feature


class domain_classifier(nn.Module):

	def __init__(self, output_dim=1):
		super(domain_classifier, self).__init__()

		self.cls = nn.Sequential(
			nn.Linear(2048, 100),
			nn.BatchNorm1d(100),
			nn.ReLU(True),
			nn.Linear(100, output_dim),
			nn.Sigmoid()
			)

	def forward(self, feature, alpha):
		reverse_feature = grad_rever_function.grad_reverse(feature, alpha)

		return self.cls(reverse_feature)

class feature_extractor(nn.Module):

	def __init__(self):
		super(feature_extractor, self).__init__()
		self.cnn = nn.Sequential(
			nn.Conv2d(3, 16, 5, 1, 2),
			nn.BatchNorm2d(16),
			nn.ReLU(True),
			nn.MaxPool2d(2),

			nn.Conv2d(16, 32, 5, 1, 2),
			nn.BatchNorm2d(32),
			nn.ReLU(True),
			nn.MaxPool2d(2),
			
			nn.Dropout2d(),
			nn.Conv2d(32, 64, 5, 1, 2),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			nn.Conv2d(64, 128, 5, 1, 2),
			nn.BatchNorm2d(128),
			nn.ReLU(True)
			)

		self.fc = nn.Sequential(
			nn.Linear(7*7*128, 1024),
			nn.BatchNorm1d(1024),
			nn.ReLU(True)
			)

	def forward(self, x):
		x = self.cnn(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x

class predictor(nn.Module):

	def __init__(self):
		super(predictor, self).__init__()
		self.fc = nn.Sequential(
			nn.Linear(1024, 1024),
			nn.BatchNorm1d(1024),
			nn.ReLU(True),
			nn.Linear(1024, 10),
			)

	def forward(self, feature, alpha=1, reverse=False):
		if reverse:
			reverse_feature = grad_rever_function.grad_reverse(feature, alpha)
			return self.fc(reverse_feature)
		return self.fc(feature)

class Identity(nn.Module):
	def __init__(self, mode='p3'):
		super(Identity, self).__init__()
		
	def forward(self, x):
		return x

class feature_extractor_1(nn.Module):

	def __init__(self):
		super(feature_extractor_1, self).__init__()

		self.cnn = models.resnet50(pretrained=True)
		self.cnn.fc = Identity()

		self.fc = nn.Sequential(
			nn.Linear(2048, 2048),
			nn.BatchNorm1d(2048),
			nn.ReLU(True),
			nn.Dropout(0.5),
			nn.Linear(2048, 345),
			)

	def forward(self, x):

		feature = self.cnn(x)
		feature = feature.view(x.size(0), -1)
		pred = self.fc(feature)

		return pred, feature

