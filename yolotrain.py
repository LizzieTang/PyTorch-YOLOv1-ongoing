# -*- coding: utf-8 -*-
# Train, backprop yolonet

__author = 'Lizzie'

import os
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision import models
import torchvision.transforms as transforms


from yolonet import YOLOnet
from yololoss import YoloLoss
from yolodata import YoloData

# from yolovisual import YoloVisual


'''
Configuration
'''
FILE_DIR = './data/VOCdevkit/VOC2012/allimgs/'
learning_rate = 0.001
num_epochs = 50
BATCH_SIZE = 4


'''
Show net structure ( Define net )
'''
print(YOLOnet)


'''
Loss Function
'''
criterion = YoloLoss()
optimizer = torch.optim.SGD(
							YOLOnet.parameters(),
							lr = learning_rate,
							momentum = 0.9,
							weight_decay = 5e-4)



'''
Train & test data ( Process inputs )
'''
trainset = YoloData(
					f_dir = FILE_DIR,
					fname_list = ['voc12_trainval.txt', 'voc07_trainval.txt'],
					tmodify = True,
					transform = [transforms.ToTensor()])
trainloader = DataLoader(
							trainset,
							batch_size = BATCH_SIZE,
							shuffle = True,
							num_workers = 2)

testset = YoloData(
					f_dir = FILE_DIR,
					fname_list = 'voc2007test.txt',
					tmodify = False,
					transform = [transforms.ToTensor()])
testloader = DataLoader(
							testset, 
							batch_size = BATCH_SIZE,
							shuffle = False,
							num_workers = 2)
print('Dataset has %d images' % (len(trainset)))
print('Batch size is %d' % BATCH_SIZE)

'''
log
'''
logfile = open('log.txt', 'w')


'''
Start Training
'''
num_iter = 0
YoloV = YoloVisual()
best_test_loss = 10.0

for epoch in range(num_epochs):

	# training per epoch
	YOLOnet.train()

	if epoch == 30:
		learning_rate = 0.0001
	if epoch == 40:
		learning_rate = 0.00001

	for param_group in optimizer.param_groups:
		param_group['lr'] = learning_rate

	print('\n\nStarting epoch %d / %d' % (epoch+1, num_epochs))
	print('Learning rate for this epoch: {}'.format(learning_rate))

	training_loss = 0.0
	for i, data in enumerate(trainloader):
		inputs, targets = data
		inputs = Variable(inputs)
		targets = Variable(targets)

		optimizer.zero_grad()
		outputs = YOLOnet(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()
		training_loss += loss.data[0]
		if (i+1) % 5 == 0:
			print('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f, average_loss: %.4f' % (epoch+1, num_epochs, 
				i+1, len(trainloader), loss.data[0], total_loss/(i+1)))
			num_iter += 1
			YoloV.plot_train_val(loss_train = training_loss/(i+1))


	# validation per epoch
	validation_loss = 0.00000
	YOLOnet.eval()
	for i, data in enumerate(testloader):
		inputs, targets = data
		inputs = Variable(inputs, volatile = True)
		targets = Variable(targets, volatile = True)

		outputs = YOLOnet(inputs)
		loss = criterion(outputs, targets)
		validation_loss += loss.data[0]
	validation_loss /= len(testloader)
	YoloV.plot_train_val(loss_val = validation_loss)

	if best_test_loss > validation_loss:
		best_test_loss = validation_loss
		print('Get best test loss %.5f' % best_test_loss)
		torch.save(net.state_dict(), 'best.pth')
	logfile.writelines(str(epoch) + '\t' + str(validation_loss) + '\n')
	logfile.flush()
	torch.save(net.state_dict(), 'yolo.pth')



