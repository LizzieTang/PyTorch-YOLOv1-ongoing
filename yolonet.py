# -*- coding: utf-8 -*-
# create the yolov1 net

__author = 'Lizzie'


import torch
import torch.nn as nn
import math


class Net(nn.Module):

	def __init__(self):
		'''
		super(),__init__() calls uper classes' __init__()
		'''
		super(Net, self).__init__()
		self.img_size = 448	# build cfg please
		#self.classes = cfg.classes	# build cfg please 
		'''
		nn.Conv2d(input_channels,out_channels,kernel_size,stride,padding,...)
		padding = x, put x to four sides of input
		example: input.size() = (1,1,4,4); padding = 1; after_padding.size() = (1,1,6,6)
		nn.MaxPool2d(kernel_size, stride, padding,...)
		'''
		'''
		P1
		'''
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)	# output of conv1 (size) = (batch_size, 64, 224, 224) 
		self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)	# output size = (batch_size, 64, 112, 112)
		'''
		P2
		'''
		self.conv2 = nn.Conv2d(64, 192, 3, 1, 1)	# (b_s, 192, 112, 112)
		self.mp2 = nn.MaxPool2d(2, 2)	# (b_s, 192, 56, 56)
		'''
		P3
		'''
		self.conv3 = nn.Conv2d(192, 128, 1, 1, 0)	# (b_s, 128, 56, 56)
		self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)	# (b_s, 256, 56, 56)
		self.conv5 = nn.Conv2d(256, 256, 1, 1, 0)	# (b_s, 256, 56, 56)
		self.conv6 = nn.Conv2d(256, 512, 3, 1, 1)	# (b_s, 512, 56, 56)
		self.mp3 = nn.MaxPool2d(2, 2)	# (b_s, 512, 28, 28)
		'''
		P4
		'''
		self.conv7 = nn.Conv2d(512, 256, 1, 1, 0)	# (b_s, 256, 28, 28)	rd=1
		self.conv8 = nn.Conv2d(256, 512, 3, 1, 1)	# (b_s, 512, 28, 28)	rd=1
		self.conv9 = nn.Conv2d(512, 256, 1, 1, 0)	# (b_s, 256, 28, 28)	rd=2
		self.conv10 = nn.Conv2d(256, 512, 3, 1, 1)	# (b_s, 512, 28, 28)	rd=2
		self.conv11 = nn.Conv2d(512, 256, 1, 1, 0)	# (b_s, 256, 28, 28)	rd=3
		self.conv12 = nn.Conv2d(256, 512, 3, 1, 1)	# (b_s, 512, 28, 28)	rd=3
		self.conv13 = nn.Conv2d(512, 256, 1, 1, 0)	# (b_s, 256, 28, 28)	rd=4
		self.conv14 = nn.Conv2d(256, 512, 3, 1, 1)	# (b_s, 512, 28, 28)	rd=4
		self.conv15 = nn.Conv2d(512, 512, 1, 1, 0)	# (b_s, 512, 28, 28)
		self.conv16 = nn.Conv2d(512, 1024, 3, 1, 1)	# (b_s, 1024, 28, 28)
		self.mp4 = nn.MaxPool2d(2, 2)	# (b_s, 1024, 14, 14)
		'''
		P5
		'''
		self.conv17 = nn.Conv2d(1024, 512, 1, 1, 0)	# (b_s, 512, 14, 14)	rd=1
		self.conv18 = nn.Conv2d(512, 1024, 3, 1, 1)	# (b_s, 1024, 14, 14)	rd=1
		self.conv19 = nn.Conv2d(1024, 512, 1, 1, 0)	# (b_s, 512, 14, 14)	rd=2
		self.conv20 = nn.Conv2d(512, 1024, 3, 1, 1)	# (b_s, 1024, 14, 14)	rd=2
		self.conv21 = nn.Conv2d(1024, 1024, 1, 1, 0)	# (b_s, 1024, 14, 14)
		self.conv22 = nn.Conv2d(1024, 1024, 3, 2, 1)	# (b_s, 1024, 7, 7)
		'''
		P6
		'''
		self.conv23 = nn.Conv2d(1024, 1024, 3, 1, 1)	# (b_s, 1024, 7, 7)
		self.conv24 = nn.Conv2d(1024, 1024, 3, 1, 1)	# (b_s, 1024, 7, 7)
		'''
		P7
		'''
		self.fc1 = nn.Linear(1024*7*7, 4096)
		'''
		P8
		'''
		self.fc2 = nn.Linear(4096, 30*7*7)
		'''
		LeakyReLU
		'''
		self.L_ReLU = nn.LeakyReLU(0.1)
		'''
		Dropout alleviate overfitting
		'''
		self.dropout = nn.Dropout(p=0.5)


	def forward(self, input_img):
		'''
		build up the network
		execpt final layer, activation functions are all Leaky ReLU
		final layer, activation function is linear activation function, i.e no activation function
		'''
		L11 = self.L_ReLU(self.conv1(input_img))
		L12 = self.mp1(L11)

		L21 = self.L_ReLU(self.conv2(L12))
		L22 = self.mp2(L21)

		L31 = self.L_ReLU(self.conv3(L22))
		L32 = self.L_ReLU(self.conv4(L31))
		L33 = self.L_ReLU(self.conv5(L32))
		L34 = self.L_ReLU(self.conv6(L33))
		L35 = self.mp3(L34)

		L41 = self.L_ReLU(self.conv7(L35))
		L42 = self.L_ReLU(self.conv8(L41))
		L43 = self.L_ReLU(self.conv9(L42))
		L44 = self.L_ReLU(self.conv10(L43))
		L45 = self.L_ReLU(self.conv11(L44))
		L46 = self.L_ReLU(self.conv12(L45))
		L47 = self.L_ReLU(self.conv13(L46))
		L48 = self.L_ReLU(self.conv14(L47))
		L49 = self.L_ReLU(self.conv15(L48))
		L410 = self.L_ReLU(self.conv16(L49))
		L411 = self.mp4(L410)

		L51 = self.L_ReLU(self.conv17(L411))
		L52 = self.L_ReLU(self.conv18(L51))
		L53 = self.L_ReLU(self.conv19(L52))
		L54 = self.L_ReLU(self.conv20(L53))
		L55 = self.L_ReLU(self.conv21(L54))
		L56 = self.L_ReLU(self.conv22(L55))

		L61 = self.L_ReLU(self.conv23(L56))
		L62 = self.L_ReLU(self.conv24(L61))

		L63 = self.dropout(self.L_ReLU(self.fc1(L62.view(L62.size(0), -1))))	# L62.size(0) = b_s

		L64 = self.fc2(L63)
		final_output = L64.view(-1, 7, 7, 30)
		return final_output

	

def __init__weights(network):

	for layer in network.modules():
		if isinstance(layer, nn.Conv2d):
			n = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
			layer.weight.data.normal_(0, math.sqrt(2.0/n))	# normalize(mean=0, std=...)
			if layer.bias is not None:
				layer.bias.data.zero_()
		elif isinstance(layer, nn.Linear):
			layer.weight.data.normal_(0, 0.001)
			layer.bias.data.zero_()




# build the net
YOLOnet = Net()
YOLOnet.apply(__init__weights) 


testNet = False
if __name__ == '__main__' and testNet:
	net = Net()
	print(net)
	input = torch.randn(1, 3, 448, 448)
	net.apply(__init__weights)
	output = net(input)
	print(output)
	print(output.size())





