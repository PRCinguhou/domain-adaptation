from torch.autograd import Function
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Function
import sys
import torch

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

class GradReverse(Function):
	@staticmethod
	def forward(ctx, x, alpha):
		ctx.alpha = alpha
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha):
	return GradReverse.apply(x, alpha)
	

def tsne_plot(cls_model, train_loader, test_loader, src_name, tar_name, batch_size, title):

	cls_model.load_state_dict(torch.load('./model/mnistm2svhn_source_only.pth'))
	cls_model.eval()
	features = []

	for index, batch in enumerate(train_loader):
		x, y = batch
		x = x.to(device)
		y = y.to(device)

		_, feature = cls_model(x)

		features.append(feature.cpu().detach().numpy())

		if index * batch_size > 2000:
			break

	for index, batch in enumerate(test_loader):
		x, _ = batch
		x = x.to(device)

		_, featrue = cls_model(x)

		features.append(featrue.cpu().detach().numpy())

		if index * batch_size > 2000:
			break

	features = np.array([featrue for featrue in features])
	features = features.reshape(-1, 2048)
	features = TSNE(n_components=2).fit_transform(features)

	plt.title('source only :' + src_name + ' to ' + tar_name)
	plt.scatter(features[:, 0], features[:, 1])
	plt.show()
	plt.savefig('./tsne_plot/'+ title + '_' + src_name + '_to_' + tar_name + '.jpg')

