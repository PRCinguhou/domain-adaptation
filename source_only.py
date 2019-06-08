import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Function
from model import encoder, domain_classifier
from LoadData import DATASET

class ToRGB(object):

	def __init__(self):
		pass
		
	def __call__(self, sample):

		sample = sample.convert('RGB')
		return sample





###		basic setting 	 ###
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
download = True
BATCH_SIZE = 100
###		-------------	 ###




mean, std = np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])

svhn_transform = transforms.Compose([
	transforms.Resize((28, 28)),
	transforms.ToTensor(),
	transforms.Normalize(mean, std)
	])

mnist_transform = transforms.Compose([
	ToRGB(),
	transforms.Resize((28, 28)),
	transforms.ToTensor(),
	transforms.Normalize(mean, std)
	])


###		 dataloader  	 ###
mnistm_train_set = dset.MNIST('./mnist', train=True, download=True, transform=mnist_transform)
mnistm_test_set = dset.MNIST('./mnist', train=False, download=True, transform=mnist_transform)


mnistm_train_loader = torch.utils.data.DataLoader(
	dataset = mnistm_train_set,
	batch_size = BATCH_SIZE,
	shuffle = True,
	)

mnistm_test_loader = torch.utils.data.DataLoader(
	dataset = mnistm_test_set,
	batch_size = BATCH_SIZE,
	shuffle = True,
	)


svhn_train_set = dset.SVHN(root='./svhn/', download=download, transform = svhn_transform)
svhn_train_loader = torch.utils.data.DataLoader(
	dataset = svhn_train_set,
	batch_size = BATCH_SIZE,
	shuffle = True,
	)

### ------------------   ###

def train(clf, optimizer, ep, train_loader, test_loader):
	#clf.load_state_dict(torch.load('./model/mnistm2svhn_source_only.pth'))
	loss_fn = nn.CrossEntropyLoss()

	for i in range(ep):

		print(i)
		for index, batch in enumerate(train_loader):

			x, y = batch
			x = x.to(device)
			y = y.to(device)
			y = y.view(-1)

			pred, _ = clf(x)

			loss = loss_fn(pred, y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		ac = 0
		for index, batch in enumerate(test_loader):

			x, y = batch
			x = x.to(device)
			y = y.to(device)

			pred, _ = clf(x)

			ac += np.sum(np.argmax(pred.cpu().detach().numpy(), axis=1) == y.cpu().detach().numpy())

		print('ac :' , ac / len(test_loader) / BATCH_SIZE)

		torch.save(clf.state_dict(), './model/mnistm2svhn_source_only.pth')


def tsne_plot(cls_model, train_loader, test_loader):

	cls_model.load_state_dict(torch.load('./model/mnistm2svhn_source_only.pth'))
	cls_model.eval()
	features = []

	for index, batch in enumerate(train_loader):
		x, y = batch
		x = x.to(device)
		y = y.to(device)

		_, feature = cls_model(x)

		features.append(feature.cpu().detach().numpy())

		if index == 20:
			break

	for index, batch in enumerate(test_loader):
		x, _ = batch
		x = x.to(device)

		_, featrue = cls_model(x)

		features.append(featrue.cpu().detach().numpy())

		if index == 20:
			break

	features = np.array([featrue for featrue in features])
	features = features.reshape(-1, 2048)
	features = TSNE(n_components=2).fit_transform(features)

	plt.scatter(features[:, 0], features[:, 1])
	plt.show()





if __name__ == '__main__':

	clf = encoder().to(device)
	optimizer = optim.Adam(clf.parameters(), lr=1e-4)

	train(clf, optimizer, 50, svhn_train_loader, mnistm_train_loader)
	tsne_plot(clf, svhn_train_loader, mnistm_train_loader)


