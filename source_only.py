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
from LoadData_1 import DATASET_1
import grad_rever_function as my_function
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
EP = 30

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

def train(clf, optimizer, ep, train_loader, test_loader, src_name, tar_name):
	#clf.load_state_dict(torch.load('./model/mnistm2svhn_source_only.pth'))
	loss_fn = nn.CrossEntropyLoss()
	ac_list, loss_list = [], []
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
		total_loss = 0
		for index, batch in enumerate(test_loader):

			x, y = batch
			x = x.to(device)
			y = y.to(device)

			pred, _ = clf(x)
			loss = loss_fn(pred, y)

			ac += np.sum(np.argmax(pred.cpu().detach().numpy(), axis=1) == y.cpu().detach().numpy())
			total_loss += loss.item()


		print('Accuracy : [%.3f], Avg Loss : [%.4f]' % ((ac / len(test_loader) / BATCH_SIZE), (total_loss / len(test_loader))) ) 
		
		ac_list.append(ac/len(test_loader)/BATCH_SIZE)
		loss_list.append(total_loss / len(test_loader) / BATCH_SIZE)

		torch.save(clf.state_dict(), './model/source_only_'+src_name+'2'+tar_name+'.pth')

	return ac_list, loss_list



def main(src, tar):

	clf = encoder().to(device)
	optimizer = optim.Adam(clf.parameters(), lr=1e-4)
		###		 dataloader  	 ###
	if src == 'mnist':
		src_train_set = dset.MNIST('./dataset/mnist', train=True, download=True, transform=gray2rgb_transform)
	
	elif src == 'mnistm':
		src_train_set = DATASET('./dataset/mnistm/train', './dataset/mnistm/train.csv', transforms=rgb_transform)
	
	elif src == 'svhn':
		src_train_set = dset.SVHN(root='./dataset/svhn/', download=download, transform=rgb_transform)

	elif src == 'usps':
		src_train_set = DATASET('./dataset/usps/train', './dataset/usps/train.csv', transforms=gray2rgb_transform)
	
	elif src == 'quickdraw':
		src_train_set = DATASET_1('quickdraw', 'quickdraw_train.csv')
	
	elif src == 'real':
		src_train_set = DATASET_1('real', 'real_train.csv')
	
	elif src == 'sketch':
		src_train_set = DATASET_1('sketch', 'sketch_train.csv')
	
	elif src == 'infograph':
		src_train_set = DATASET_1('infograph', 'infograph_train.csv')
	



	if tar == 'svhn':
		tar_train_set = dset.SVHN(root='./dataset/svhn/', download=download, transform = rgb_transform)
	
	elif tar == 'mnist':
		tar_train_set = dset.MNIST('./dataset/mnist', train=True, download=True, transform=gray2rgb_transform)
	
	elif tar == 'mnistm':
		tar_train_set = DATASET('./dataset/mnistm/train', './dataset/mnistm/train.csv', transform=rgb_transform)

	elif tar == 'usps':
		tar_train_set = DATASET('./dataset/usps/train', './dataset/usps/train.csv', transform=rgb_transform)

	elif tar == 'infograph':
		tar_train_set = DATASET_1('infograph', 'infograph_train.csv')
	
	elif tar == 'sketch':
		tar_train_set = DATASET_1('sketch', 'sketch_train.csv')
	
	elif tar == 'quickdraw':
		tar_train_set = DATASET_1('quickdraw', 'quickdraw_train.csv')
	
	elif tar == 'real':
		tar_train_set = DATASET_1('real', 'real_train.csv')
	

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
	ac_list, loss_list = train(clf, optimizer, EP, src_train_loader, tar_train_loader, src, tar)
	ac_list = np.array(ac_list)
	loss_list = np.array(loss_list)
	epoch = [i for i in range(EP)]
	
	# plot tsne
	my_function.tsne_plot(clf, src_train_loader, tar_train_loader, src, tar, BATCH_SIZE, 'source_only')

	### plot learning curve  ###
	plt.figure()
	plt.plot(epoch, ac_list)
	plt.xlabel('EPOCH')
	plt.ylabel('Accuracy')
	plt.title('Source_only : ' + src + ' to ' + tar)
	plt.savefig('./learning_curve/source_only_' + src + '_to_' + tar + '_accuracy.jpg')

	plt.figure()
	plt.plot(epoch, loss_list)
	plt.xlabel('EPOCH')
	plt.ylabel('Loss')
	plt.title('Source_only : ' + src + ' to ' + tar)
	plt.savefig('./learning_curve/source_only_' + src + '_to_' + tar + '_loss.jpg')
	


if __name__ == '__main__':

	source, target = sys.argv[1:]
	main(source, target)


