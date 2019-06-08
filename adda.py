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



def train(src_model, tar_model, domain_clf, optimizer_domain, optimizer_tar, ep, train_loader, test_loader, src_name, tar_name):

	loss_fn_cls = nn.CrossEntropyLoss()
	loss_fn_domain = nn.MSELoss()
	
	ac_list = []
	loss_list = []
	for i in range(ep):

		src_model.train()
		tar_model.train()
		domain_clf.train()

		for index, (src_batch, tar_batch) in enumerate(zip(train_loader, test_loader)):

			optimizer_domain.zero_grad()

			src_x, src_y = src_batch
			tar_x, _ = tar_batch

			src_x = src_x.to(device)
			src_y = src_y.to(device)
			src_y = src_y.view(-1)

			tar_x = tar_x.to(device)

			_, src_feature = src_model(src_x)
			_, tar_feature = tar_model(tar_x)

			src_domain_pred = domain_clf(src_feature)
			src_domain_label = torch.ones(src_x.size(0))

			tar_domain_pred = domain_clf(tar_feature)
			tar_domain_label = torch.zeros(tar_x.size(0))

			domain_loss = loss_fn_domain(src_domain_pred, src_domain_label) + loss_fn_domain(tar_domain_pred, tar_domain_label)

			domain_loss.backward()
			optimizer_domain.step()



			optimizer_tar.zero_grad()

			_, tar_feature = tar_model(tar_x)
			tar_domain_pred = domain_clf(tar_feature)
			tar_domain_label = torch.ones(tar_x.size(0))

			loss = loss_fn_domain(tar_domain_pred, tar_domain_label)
			loss.backward()
			optimizer_tar.step()

			if index % 100 == 0:
				print('EP : [%d], [%d]/[%d]' % (i, index, min(len(train_loader), len(test_loader))))

		src_model.eval()
		tar_model.eval()
		domain_clf.eval()

		ac = 0
		avg_loss = 0

		with torch.no_grad():
			for index, batch in enumerate(test_loader):

				x, y = batch
				x = x.to(device)
				y = y.to(device)
				y = y.view(-1)

				pred, _ = tar_model(x)

				loss = loss_fn_cls(pred, y)

				avg_loss += loss.item()
				ac += np.sum(np.argmax(pred.cpu().detach().numpy(), axis=1) == y.cpu().detach().numpy())

		print("AVG loss : [%.4f]" % (avg_loss/len(test_loader)))
		print('Accuracy : [%.4f]' % (ac / len(test_loader) / BATCH_SIZE))


def main(src, tar):


	src_clf = encoder().to(device)
	src_clf.load_state_dict(torch.load('./model/reverse_grad_'+src+'2'+tar+'.pth'))

	tar_clf = encoder().to(device)

	domain_clf = domain_classifier().to(device)
	optimizer_domain = optim.Adam(domain_clf.parameters(), lr=1e-4)
	optimizer_tar = optim.Adam(tar_clf.parameters(), lr=1e-4)


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
	ac_list, loss_list = train(clf, domain_clf, optimizer_domain, optimizer_tar, 50, src_train_loader, tar_train_loader)
	ac_list = np.array(ac_list)
	
	# plot tsne
	loss_list = np.array(loss_list)
	epoch = [i for i in range(EP)]
	my_function.tsne_plot(clf, domain_clf, src_train_loader, tar_train_loader, src, tar, BATCH_SIZE, 'adda_')

	### plot learning curve  ###
	plt.plot(ac_list, epoch)
	plt.xlabel('EPOCH')
	plt.ylabel('Accuracy')
	plt.title('domian_adapt : ' + src + ' to ' + tar)
	plt.savefig('./learning_curve/adda_' + src + '_to_' + tar + '_accuracy.jpg')

	plt.plot(loss_list, epoch)
	plt.xlabel('EPOCH')
	plt.ylabel('Loss')
	plt.title('domian_adapt : ' + src + ' to ' + tar)
	plt.savefig('./learning_curve/adda_' + src + '_to_' + tar + '_loss.jpg')
	


if __name__ == '__main__':

	source, target = sys.argv[1:]
	main(source, target)


