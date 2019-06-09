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
BATCH_SIZE = 256
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

clf = encoder().to(device)
domain_clf = domain_classifier().to(device)
optimizer = optim.Adam(list(clf.parameters()) + list(domain_clf.parameters()) , lr=1e-4)


def train(cls_model, domain_clf, optimizer, ep, train_loader, test_loader, src_name, tar_name):
	loss_fn_cls = nn.CrossEntropyLoss()
	loss_fn_domain = nn.MSELoss()
	ac_list, loss_list = [], []

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

			src_pred, src_feature = clf(x)
			_, tar_feature = clf(tar_x)

			label_loss = loss_fn_cls(src_pred, y)
			src_domain = domain_clf(src_feature, alpha)
			tar_domain = domain_clf(tar_feature, alpha)

			src_domain_label = torch.ones(x.size(0)).to(device)
			tar_domain_label = torch.zeros(tar_x.size(0)).to(device)
			domain_loss = loss_fn_domain(src_domain, src_domain_label) + loss_fn_domain(tar_domain, tar_domain_label)
			cls_loss = loss_fn_cls(src_pred, y)

			loss = cls_loss + domain_loss


			optimizer.zero_grad()
			loss.backward()
			optimizer.step()



		cls_model.eval()
		domain_clf.eval()

		ac = 0
		total_loss=0
		with torch.no_grad():
			for batch in test_loader:
				
				x, y = batch
				x = x.to(device)
				y = y.to(device)
				y = y.view(-1)

				pred, _ = clf(x)
				loss = loss_fn_cls(pred, y)

				ac += np.sum(np.argmax(pred.cpu().detach().numpy(), axis=1) == y.cpu().detach().numpy())
				total_loss += loss.item()
		
		print('Accuracy : [%.3f], Avg Loss : [%.4f]' % ((ac / len(test_loader) / BATCH_SIZE), (total_loss / len(test_loader))) ) 
	
		ac_list.append(ac/len(test_loader)/BATCH_SIZE)
		loss_list.append(total_loss / len(test_loader) / BATCH_SIZE)

		torch.save(clf.state_dict(), './model/reverse_grad_'+src_name+'2'+tar_name+'.pth')
	return ac_list, loss_list




def main(src, tar):
	###		 dataloader  	 ###
	if src == 'mnist':
		src_train_set = dset.MNIST('./dataset/mnist', train=True, download=True, transform=gray2rgb_transform)
	
	elif src == 'mnistm':
		src_train_set = DATASET('./dataset/mnistm/train', './dataset/mnistm/train.csv', transforms=rgb_transform)
	
	elif src == 'svhn':
		src_train_set = dset.SVHN(root='./dataset/svhn/', download=download, transform=rgb_transform)


	if tar == 'svhn':
		tar_train_set = dset.SVHN(root='./dataset/svhn/', download=download, transform = rgb_transform)
	
	elif tar == 'mnist':
		tar_train_set = dset.MNIST('./dataset/mnist', train=True, download=True, transform=gray2rgb_transform)
	
	elif tar == 'mnistm':
		src_train_set = DATASET('./dataset/mnistm/train', './dataset/mnistm/train.csv', transform=rgb_transform)
	


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
	ac_list, loss_list = train(clf, domain_clf, optimizer, EP, src_train_loader, tar_train_loader, src, tar)
	ac_list = np.array(ac_list).flatten()
	
	# plot tsne
	loss_list = np.array(loss_list).flatten()
	epoch = [i for i in range(EP)]
	my_function.tsne_plot(clf, src_train_loader, tar_train_loader, src, tar, BATCH_SIZE, 'reverse_grad')

	### plot learning curve  ###
	plt.figure()
	plt.plot(epoch, ac_list)
	plt.xlabel('EPOCH')
	plt.ylabel('Accuracy')
	plt.title('domian_adapt : ' + src + ' to ' + tar)
	plt.savefig('./learning_curve/domian_adapt_' + src + '_to_' + tar + '_accuracy.jpg')

	plt.figure()
	plt.plot(epoch, loss_list)
	plt.xlabel('EPOCH')
	plt.ylabel('Loss')
	plt.title('domian_adapt : ' + src + ' to ' + tar)
	plt.savefig('./learning_curve/domian_adapt_' + src + '_to_' + tar + '_loss.jpg')
	


if __name__ == '__main__':

	source, target = sys.argv[1:]
	main(source, target)


