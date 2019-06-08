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
import grad_rever_function as my_function


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

rgb_transform = transforms.Compose([
	transforms.Resize((28, 28)),
	transforms.ToTensor(),
	transforms.Normalize(mean, std)
	])

gray2rgb_transform = transforms.Compose([
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

			p = float(index + i * min([len(train_loader), len(test_loader)])) / ep / min([len(train_loader), len(test_loader)])
			alpha = 2. / (1. + np.exp(-10 * p)) - 1

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

			domain_pred = domain_clf(feature, alpha)
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





def main(src, tar):
	###		 dataloader  	 ###
	if src == 'mnist':
		src_train_set = dset.MNIST('./mnist', train=True, download=True, transform=gray2rgb_transform)
	
	elif src == 'mnistm':
		src_train_set = DATASET('./mnistm/train', './mnistm/train.csv', transforms=rgb_transform)
	
	elif src == 'svhn':
		src_train_set = dset.SVHN(root='./svhn/', download=download, transform=rgb_transform)


	if tar == 'svhn':
		tar_train_set = dset.SVHN(root='./svhn/', download=download, transform = rgb_transform)
	
	elif tar == 'mnist':
		tar_train_set = dset.MNIST('./mnist', train=True, download=True, transform=gray2rgb_transform)
	
	elif tar == 'mnistm':
		src_train_set = DATASET('./mnistm/train', './mnistm/train.csv', transform=rgb_transform)
	


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

	# train
	ac_list, loss_list = train(clf, domain_clf, optimizer, 50, src_train_loader, tar_train_loader)
	ac_list = np.array(ac_list)
	
	# plot tsne
	loss_list = np.array(loss_list)
	epoch = [i for i in range(EP)]
	my_function.tsne_plot(clf, domain_clf, src_train_loader, tar_train_loader, src, tar, BATCH_SIZE, 'domian_adapt')

	### plot learning curve  ###
	plt.plot(ac_list, epoch)
	plt.xlabel('EPOCH')
	plt.ylabel('Accuracy')
	plt.title('domian_adapt : ' + src + ' to ' + tar)
	plt.savefig('./learning_curve/domian_adapt_' + src + '_to_' + tar + '_accuracy.jpg')

	plt.plot(loss_list, epoch)
	plt.xlabel('EPOCH')
	plt.ylabel('Loss')
	plt.title('domian_adapt : ' + src + ' to ' + tar)
	plt.savefig('./learning_curve/domian_adapt_' + src + '_to_' + tar + '_loss.jpg')
	


if __name__ == '__main__':

	source, target = sys.argv[1:]
	main(source, target)


