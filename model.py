import torch.nn as nn
import torch
import numpy as np
import grad_rever_function

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

