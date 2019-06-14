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
from model import encoder, predictor
from LoadData import DATASET
import sys
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

'''
dataset : infograph, quickdraw, real, sketch
'''

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
#print('cuda = ', cuda)
BATCH_SIZE = 256
EP = 50

class ToRGB(object):

	def __init__(self):
		pass
		
	def __call__(self, sample):

		sample = sample.convert('RGB')
		return sample

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])

transform = transforms.Compose([
	ToRGB(),
	transforms.Resize((32, 32)),
	transforms.ToTensor(),
	transforms.Normalize(mean, std)
	])

if __name__ == '__main__':
	
	argument = sys.argv[1:]
	source_domain = argument[:-1]
	target_domain = argument[-1]
	N = len(source_domain)
	# dataloader
	source_dataloader_list = []
	source_clf = {}

	extractor = encoder().to(device)
	extractor_optim = optim.Adam(extractor.parameters(), lr=3e-4)

	for source in source_domain:
		print(source)
		if source == 'svhn':
			dataset = dset.SVHN(root='./dataset/svhn/', download=True, transform=transform)
		elif source == 'mnist':
			dataset = dset.MNIST('./dataset/mnist', train=True, download=True, transform=transform)
		else:
			print(source)
			dataset = DATASET(source, 'train')
		dataset = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle=True)
		source_dataloader_list.append(dataset)

		# c1 : for target
		# c2 : for source
		source_clf[source] = {}
		source_clf[source]['c1'] = predictor().to(device)
		source_clf[source]['c2'] = predictor().to(device)
		source_clf[source]['optim'] = optim.Adam(list(source_clf[source]['c1'].parameters()) + list(source_clf[source]['c2'].parameters()), lr=3e-4, weight_decay=0.0005)
	
	if target_domain == 'svhn':
		target_dataset = dset.SVHN(root='./dataset/svhn/', download=True, transform=transform)
	elif target_domain == 'mnist':
		target_dataset = dset.MNIST('./dataset/mnist', train=True, download=True, transform=transform)
	else:
		target_dataset = DATASET(target_domain, 'train')

	target_dataloader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=True)

	loss_extractor = nn.CrossEntropyLoss()


	for ep in range(EP):
		
		print(ep+1)

		extractor.train()

		source_ac = {}
		for source in source_domain:
			source_clf[source]['c1'] = source_clf[source]['c1'].train()
			source_clf[source]['c2'] = source_clf[source]['c2'].train()
			source_ac[source] = defaultdict(int)


		for batch_index, (src_batch, tar_batch) in enumerate(zip(zip(*source_dataloader_list), target_dataloader)):
			

			src_len = len(src_batch)
			loss_cls = 0
			# train extractor and source clssifier
			for index, batch in enumerate(src_batch):
				x, y = batch
				x = x.to(device)
				y = y.to(device)
				y = y.view(-1)

				feature = extractor(x)
				pred1 = source_clf[source_domain[index]]['c1'](feature)
				pred2 = source_clf[source_domain[index]]['c2'](feature)

				source_ac[source_domain[index]]['c1'] += torch.sum(torch.max(pred1, dim=1)[1] == y).item()
				source_ac[source_domain[index]]['c2'] += torch.sum(torch.max(pred2, dim=1)[1] == y).item()
				loss_cls += loss_extractor(pred1, y) + loss_extractor(pred2, y)


			if batch_index % 5 == 0:
				for source in source_domain:
						print(source)
						print('c1 : [%.8f]' % (source_ac[source]['c1']/(batch_index+1)/BATCH_SIZE))
						print('c2 : [%.8f]' % (source_ac[source]['c2']/(batch_index+1)/BATCH_SIZE))
					
						print('\n')

				

			#extractor_optim.zero_grad()
			#for index, source in enumerate(source_domain):
			#	source_clf[source_domain[index]]['optim'].zero_grad()
			
			#loss_cls.backward(retain_graph=True)

			#extractor_optim.step()
			
			#for index, source in enumerate(source_domain):	
			#	source_clf[source]['optim'].step()		
			#	source_clf[source]['optim'].zero_grad()

			#extractor_optim.zero_grad()
			


			m1_loss = 0
			m2_loss = 0
			for k in range(1, 3):
				for i_index, batch in enumerate(src_batch):
					x, y = batch
					x = x.to(device)
					y = y.to(device)
					y = y.view(-1)

					tar_x, _ = tar_batch
					tar_x = tar_x.to(device)

					src_feature = extractor(x)
					tar_feature = extractor(tar_x)

					e_src = torch.mean(src_feature**k, dim=0)
					e_tar = torch.mean(tar_feature**k, dim=0)
					m1_dist = e_src.dist(e_tar)
					m1_loss += m1_dist
					for j_index, other_batch in enumerate(src_batch[i_index:]):
						other_x, other_y = other_batch
						other_x = other_x.to(device)
						other_y = other_y.to(device)
						other_y = other_y.view(-1)
						other_feature = extractor(other_x)

						e_other = torch.mean(other_feature**k, dim=0)
						m2_dist = e_src.dist(e_other)
						m2_loss += m2_dist
						
			loss_m =  0.5 * (m1_loss/N + m2_loss/N/(N-1)*2) 
			
			loss = loss_cls 

			if batch_index % 5 == 0:
				print('[%d]/[%d]' % (batch_index, len(target_dataloader)))
				print('class loss : [%.5f]' % (loss_cls))
				print('msd loss : [%.5f]' % (loss_m))

			
			extractor_optim.zero_grad()
			for source in source_domain:
				source_clf[source]['optim'].zero_grad()
			
			loss.backward(retain_graph=True)

			extractor_optim.step()
			
			for source in source_domain:	
				source_clf[source]['optim'].step()	
				source_clf[source]['optim'].zero_grad()
				

			extractor_optim.zero_grad()
			

			tar_x , _ = tar_batch
			tar_x = tar_x.to(device)
			tar_feature = extractor(tar_x)
			loss = 0

			for index, batch in enumerate(src_batch):
				x, y = batch
				x = x.to(device)
				y = y.to(device)
				y = y.view(-1)

				feature = extractor(x)

				pred1 = source_clf[source_domain[index]]['c1'](feature)
				pred2 = source_clf[source_domain[index]]['c2'](feature)

				clf_loss = loss_extractor(pred1, y) + loss_extractor(pred2, y)
				
				pred_c1 = source_clf[source_domain[index]]['c1'](tar_feature)
				pred_c2 = source_clf[source_domain[index]]['c2'](tar_feature)
				
				discrepency_loss = torch.mean(torch.sum(abs(F.softmax(pred_c1, dim=1) - F.softmax(pred_c2, dim=1)), dim=1))

				loss += clf_loss - discrepency_loss
				
			loss.backward(retain_graph=True)
			for source in source_domain:
				source_clf[source]['optim'].zero_grad()
				source_clf[source]['optim'].step()
				source_clf[source]['optim'].zero_grad()
				extractor_optim.zero_grad()

			discrepency_loss = 0
			for index, _ in enumerate(src_batch):

				pred_c1 = source_clf[source_domain[index]]['c1'](tar_feature)
				pred_c2 = source_clf[source_domain[index]]['c2'](tar_feature)
				
				discrepency_loss += torch.mean(torch.sum(abs(F.softmax(pred_c1, dim=1) - F.softmax(pred_c2, dim=1)), dim=1))

			extractor_optim.zero_grad()
			discrepency_loss.backward(retain_graph=True)
			extractor_optim.step()
			extractor_optim.zero_grad()
			for source in source_domain:
				source_clf[source]['optim'].zero_grad()
			
			if batch_index % 5 == 0:
				print('Discrepency Loss : [%.4f]' % (discrepency_loss))

		extractor.eval()
		for source in source_domain:
			source_clf[source]['c1'] = source_clf[source]['c1'].eval()
			source_clf[source]['c2'] = source_clf[source]['c2'].eval()
		
		source_ac = {}

		if target_domain == 'svhn':
			eval_loader = dset.SVHN(root='./dataset/svhn/', download=True, transform=transform)
		elif target_domain == 'mnist':
			eval_loader = dset.MNIST('./dataset/mnist', train=True, download=True, transform=transform)
		else:
			eval_loader = DATASET(target_domain, 'train')
		eval_loader = DataLoader(eval_loader, batch_size=BATCH_SIZE, shuffle=True)

		for source in source_domain:
			source_ac[source] = defaultdict(int)
		fianl_ac = 0


		with torch.no_grad():
			for index, batch in enumerate(eval_loader):
				x, y = batch
				x = x.to(device)
				y = y.to(device)
				y = y.view(-1)

				feature = extractor(x)
				final_pred = 1
				for source in source_domain:
					pred1 = source_clf[source]['c1'](feature)
					pred2 = source_clf[source]['c2'](feature)
					if isinstance(final_pred, int):
						final_pred = F.softmax(pred1, dim=1) + F.softmax(pred2, dim=1)	
					else:
						final_pred += F.softmax(pred1, dim=1) + F.softmax(pred2, dim=1)	
					

					source_ac[source]['c1'] += np.sum(np.argmax(pred1.cpu().detach().numpy(), axis=1) == y.cpu().detach().numpy())
					source_ac[source]['c2'] += np.sum(np.argmax(pred2.cpu().detach().numpy(), axis=1) == y.cpu().detach().numpy())
				fianl_ac += np.sum(np.argmax(final_pred.cpu().detach().numpy(), axis=1) == y.cpu().detach().numpy())
		for source in source_domain:
			print('Current Source : ', source)
			print('Accuray for c1 : [%.4f]' % (source_ac[source]['c1']/BATCH_SIZE/len(eval_loader)))
			print('Accuray for c2 : [%.4f]' % (source_ac[source]['c2']/BATCH_SIZE/len(eval_loader)))
		print('Combine Ac : [%.4f]' % (fianl_ac/BATCH_SIZE/len(eval_loader)))

		torch.save(extractor.state_dict(), './model/extractor'+'_'+str(ep)+'.pth')
		for source in source_domain:
			torch.save(source_clf[source]['c1'].state_dict(), './model/'+source+'_c1_'+str(ep)+'.pth')
			torch.save(source_clf[source]['c2'].state_dict(), './model/'+source+'_c2_'+str(ep)+'.pth')
						



				






















