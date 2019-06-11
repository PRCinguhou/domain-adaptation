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
from model import encoder, domain_classifier, feature_extractor_1
from LoadData import DATASET
from LoadData_1 import DATASET_1
import sys
import grad_rever_function as my_function

import argparse
import os
import random
import shutil
import time
import warnings

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
BATCH_SIZE = 30
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


def train(cls_model, domain_clf, optimizer, ep, train_loader, test_loader, src_name, tar_name):
	loss_fn_cls = nn.CrossEntropyLoss()
	loss_fn_domain = nn.MSELoss()
	ac_list, loss_list = [], []
	max_= 0

	batch_time = AverageMeter('Time', ':6.3f')
	data_time = AverageMeter('Data', ':6.3f')
	losses = AverageMeter('Loss', ':.4e')
	top1 = AverageMeter('Acc@1', ':6.2f')
	top5 = AverageMeter('Acc@5', ':6.2f')
	progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
							 top5, prefix="Epoch: [{}]".format(ep))

	end = time.time()

	for i in range(ep):

		cls_model.train()
		domain_clf.train()
		print(i)
		for index, (src_batch, tar_batch) in enumerate(zip(train_loader, test_loader)):
			if index % 10 == 0:
				print('start')
			data_time.update(time.time() - end)

				
			p = float(index + i * min([len(train_loader), len(test_loader)])) / ep / min([len(train_loader), len(test_loader)])
			alpha = 2. / (1. + np.exp(-10 * p)) - 1
			alpha = 1
			
			x, y = src_batch
			x = x.to(device)
			y = y.to(device)
			y = y.view(-1)
			tar_x, _ = tar_batch
			tar_x = tar_x.to(device)

			src_pred, src_feature = cls_model(x)
			_, tar_feature = cls_model(tar_x)

			label_loss = loss_fn_cls(src_pred, y)
			losses.update(label_loss.item(), x.size(0))

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

			if index % 10 == 0:
				print('[%d]/[%d]' % (index, min([len(train_loader), len(test_loader)])))
			batch_time.update(time.time() - end)
			
			end = time.time()

			if index % 10 == 0:
				progress.print(i)


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

				pred, _ = cls_model(x)
				loss = loss_fn_cls(pred, y)

				ac += np.sum(np.argmax(pred.cpu().detach().numpy(), axis=1) == y.cpu().detach().numpy())
				total_loss += loss.item()
		
		print('Accuracy : [%.3f], Avg Loss : [%.4f]' % ((ac / len(test_loader) / BATCH_SIZE), (total_loss / len(test_loader))) ) 
		
		ac_list.append(ac/len(test_loader)/BATCH_SIZE)
		loss_list.append(total_loss / len(test_loader) / BATCH_SIZE)
		if (ac / len(test_loader) / BATCH_SIZE) > max_:
			max_ = (ac / len(test_loader) / BATCH_SIZE)
			torch.save(cls_model.state_dict(), './model/reverse_grad_'+src_name+'2'+tar_name+'.pth')
	return ac_list, loss_list




def main(src, tar):

	
	domain_clf = domain_classifier().to(device)
	optimizer = optim.Adam(list(clf.parameters()) + list(domain_clf.parameters()) , lr=1e-4)

	###		 dataloader  	 ###
	if src == 'mnist':
		src_train_set = dset.MNIST('./dataset/mnist', train=True, download=True, transform=gray2rgb_transform)
		clf = encoder().to(device)
		
	elif src == 'mnistm':
		src_train_set = DATASET('./dataset/mnistm/train', './dataset/mnistm/train.csv', transforms=rgb_transform)
		clf = encoder().to(device)

	elif src == 'svhn':
		src_train_set = dset.SVHN(root='./dataset/svhn/', download=download, transform=rgb_transform)
		clf = encoder().to(device)

	elif src == 'usps':
		src_train_set = DATASET('./dataset/usps/train', './dataset/usps/train.csv', transforms=gray2rgb_transform)
		clf = encoder().to(device)

	elif src == 'quickdraw':
		src_train_set = DATASET_1('quickdraw', 'quickdraw_train.csv')
		clf = feature_extractor_1().to(device)

	elif src == 'real':
		src_train_set = DATASET_1('real', 'real_train.csv')
		clf = feature_extractor_1().to(device)

	elif src == 'sketch':
		src_train_set = DATASET_1('sketch', 'sketch_train.csv')
		clf = feature_extractor_1().to(device)

	elif src == 'infograph':
		src_train_set = DATASET_1('infograph', 'infograph_train.csv')
		clf = feature_extractor_1().to(device)




	if tar == 'svhn':
		tar_train_set = dset.SVHN(root='./dataset/svhn/', download=download, transform = rgb_transform)
		clf = encoder().to(device)

	elif tar == 'mnist':
		tar_train_set = dset.MNIST('./dataset/mnist', train=True, download=True, transform=gray2rgb_transform)
		clf = encoder().to(device)

	elif tar == 'mnistm':
		tar_train_set = DATASET('./dataset/mnistm/train', './dataset/mnistm/train.csv', transform=rgb_transform)
		clf = encoder().to(device)

	elif tar == 'usps':
		tar_train_set = DATASET('./dataset/usps/train', './dataset/usps/train.csv', transform=rgb_transform)
		clf = encoder().to(device)
		
	elif tar == 'infograph':
		tar_train_set = DATASET_1('infograph', 'infograph_train.csv')
		clf = feature_extractor_1().to(device)

	elif tar == 'sketch':
		tar_train_set = DATASET_1('sketch', 'sketch_train.csv')
		clf = feature_extractor_1().to(device)

	elif tar == 'quickdraw':
		tar_train_set = DATASET_1('quickdraw', 'quickdraw_train.csv')
		clf = feature_extractor_1().to(device)

	elif tar == 'real':
		tar_train_set = DATASET_1('real', 'real_train.csv')
		clf = feature_extractor_1().to(device)



	src_train_loader = torch.utils.data.DataLoader(
		dataset = src_train_set,
		batch_size = BATCH_SIZE,
		shuffle = True,
		pin_memory=True
		)

	tar_train_loader = torch.utils.data.DataLoader(
		dataset = tar_train_set,
		batch_size = BATCH_SIZE,
		shuffle = True,
		pin_memory=True
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
	

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self, name, fmt=':f'):
		self.name = name
		self.fmt = fmt
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

	def __str__(self):
		fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
		return fmtstr.format(**self.__dict__)
		
class ProgressMeter(object):
	def __init__(self, num_batches, *meters, prefix=""):
		self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
		self.meters = meters
		self.prefix = prefix

	def print(self, batch):
		entries = [self.prefix + self.batch_fmtstr.format(batch)]
		entries += [str(meter) for meter in self.meters]
		print('\t'.join(entries))

	def _get_batch_fmtstr(self, num_batches):
		num_digits = len(str(num_batches // 1))
		fmt = '{:' + str(num_digits) + 'd}'
		return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':

	source, target = sys.argv[1:]
	main(source, target)


