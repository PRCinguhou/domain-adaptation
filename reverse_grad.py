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
import sys

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





clf = encoder().to(device)
domain_clf = domain_classifier().to(device)
optimizer = optim.Adam(list(clf.parameters()) + list(domain_clf.parameters()) , lr=1e-4)


def train(cls_model, domain_clf, optimizer, ep, train_loader, test_loader):
	loss_fn_cls = nn.CrossEntropyLoss()
	loss_fn_domain = nn.MSELoss()

	for i in range(ep):

		cls_model.train()
		for index, (src_batch, tar_batch) in enumerate(zip(train_loader, test_loader)):

			x, y = src_batch
			x = x.to(device)
			y = y.to(device)
			y = y.view(-1)
			tar_x, _ = tar_batch
			tar_x = tar_x.to(device)

			input_x = torch.cat([x, tar_x]).to(device)

			domain_y = torch.cat([torch.ones(x.size(0)), torch.zeros(tar_x.size(0))]).to(device)

			label_pred, feature = cls_model(input_x)
			label_pred = label_pred[:x.size(0)]

			domain_pred = domain_clf(feature)
			domain_loss = loss_fn_domain(domain_pred, domain_y)
			cls_loss = loss_fn_cls(label_pred, y)

			loss = domain_loss + cls_loss

			if index % 100 == 0:
				print("[%d/%d]" % (index, len(train_loader)))
				print('cls loss : [%.5f]' % (cls_loss.item()))
				print('domain loss : [%.5f]' % (domain_loss.item()))

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()



		cls_model.eval()
		domain_clf.eval()
		ac = 0
		with torch.no_grad():
			for batch in test_loader:
				
				x, y = batch
				x = x.to(device)
				y = y.to(device)
				y = y.view(-1)
				y = y.view(-1)

				pred, _ = cls_model(x)

				ac += np.sum(np.argmax(pred.cpu().detach().numpy(), axis = 1 ) == y.cpu().detach().numpy())

			print('ac :' , ac / len(test_loader) / BATCH_SIZE)

		torch.save(clf.state_dict(), './model/mnistm2svhn.pth')

def test(cls_model, domain_clf, train_loader, test_loader):

	cls_model.load_state_dict(torch.load('./model/mnistm2svhn.pth'))
	cls_model.eval()
	domain_clf.eval()
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





def main(src, tar):
	###		 dataloader  	 ###
	if src == 'mnist':
		src_train_set = dset.MNIST('./mnist', train=True, download=True, transform=mnist_transform)
	
	elif src == 'mnistm':
		src_train_set = DATASET('./mnistm/train', './mnistm/train.csv')
	
	elif src == 'svhn':
		src_train_set = dset.SVHN(root='./svhn/', download=download, transform = svhn_transform)


	if tar == 'svhn':
		tar_train_set = dset.SVHN(root='./svhn/', download=download, transform = svhn_transform)
	
	elif tar == 'mnist':
		tar_train_set = dset.MNIST('./mnist', train=True, download=True, transform=mnist_transform)
	
	elif tar == 'mnistm':
		src_train_set = DATASET('./mnistm/train', './mnistm/train.csv')
	


	src_train_loader = torch.utils.data.DataLoader(
		dataset = src_train_set,
		batch_size = BATCH_SIZE,
		shuffle = True,
		)

	tar_train_loader = torch.utils.data.DataLoader(
		dataset = tar_train_set,
		batch_size = BATCH_SIZE,
		shuffle = True,
		)

	### ------------------   ###

	train(clf, domain_clf, optimizer, 50, src_train_loader, tar_train_loader)
	test(clf, domain_clf, src_train_loader, tar_train_loader)

if __name__ == '__main__':
	source, target = sys.argv[1:]
	main(source, target)








