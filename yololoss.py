# -*- coding: utf-8 -*-
# 3. Loss function
# lossfunction refers to yolo1 thesis and xiongzihua

__author = 'Lizzie'

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class YoloLoss(nn.Module):

	def __init__(self):
		'''
		args:
			S = 7 : 7*7 grids
			B = 2 : 2 bboxes per grid
			l_coord >> lambda_coord = 5.0 in yolo1 thesis
			l_noobj >> lambda_noobj = 0.5 in yolo1 thesis
		'''
		super(YoloLoss, self).__init__()
		self.S = 7
		self.B = 2
		self.l_coord = 5.0
		self.l_noobj = 0.5


	def compute_iou(self, box1, box2):
		'''
		IOU == intersection(box1, box2) / union(box1, box2)
		args:
			box1.size() = (N, 4)
			box2.size() = (M, 4)
			box1[i] or box2[i] = tensor([x1, y1, x2, y2])
		return:
			tensor.size() = (N, M)
		'''
		N = box1.size(0)	# take batch size of box1 N as column number of ouput
		M = box2.size(0)	# take batch size of box2 M as row number of ouput
		'''
		torch1.size() = (m, n)
		0  1  2
		 (m, n)
		torch1.unsqueeze(0).size() = (1, m, n)
		torch1.unsqueeze(1).size() = (m, 1, n)
		torch1.unsqueeze(2).size() = (m, n, 1)
		'''
		lt = torch.max(
						box1[:, :2].unsqueeze(1).expand(N, M, 2),	# box1[:, :2].size() : (N, 2) -> (N, 1, 2) -> (N, M, 2) >> M columns are the same
						box2[:, :2].unsqueeze(0).expand(N, M, 2)	# box2[:, :2].size() : (M, 2) -> (1, M, 2) -> (N, M, 2) >> N rows are the same
					  )	# select the left top point of the intersection, lt.size() = (N, M, 2) >> N columns are different, M rows are different

		rb = torch.min(
						box1[:, 2:].unsqueeze(1).expand(N, M, 2),
						box2[:, 2:].unsqueeze(0).expand(N, M, 2)
					  )	# select the right bottom point of the intersection

		wh = rb - lt 	# wh.size() = (N, M, 2) >> width and height of intersection
		wh[wh<0] = 0	# wh < 0 means box1 and box2 have no intersection
		interarea = wh[:, :, 0] * wh[:, :, 1]	# interarea.size() = (N, M)
		# interarea[i, j] = the area of intersection of the i box(box1) and j box(box2)

		area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])	# the area of box1, area1.size() = (N,)
		area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])	# the area of box2, area2.size() = (M,)
		area1 = area1.unsqueeze(1).expand_as(interarea)	# area1.size() : (N,) -> (N, 1) -> (N, M) >> M columns are the same
		area2 = area2.unsqueeze(0).expand_as(interarea)	# area2.size() : (M,) -> (1, M) -> (N, M) >> N rows are the same

		iou = interarea / (area1 + area2 - interarea)

		return iou


	def forward(self, pred_tensor, target_tensor):
		'''
		pred_tensor.size() = (batchsize, 7, 7, 30)
		target_tensor.size() = (batchsize, 7, 7, 30)
		30 = 2*[x, y, w, h, c] + 20*classes, pred_tensor.size() == target_tensor.size()
		args:
			x: (0, 1),  x corrdinate of one grid, 
			y: (0, 1),  y corrdinate of one grid,
			w: (0, 1),  width of bbox == (pixel bbox width)/(pixel image width)
			h: (0, 1),  height of bbox == (pixel bbox height)/(pixel image height)
			example: [0.5, 0.5, 0.5, 0.5] in a 448*448 image, center of bbox is the center of the grid, pixel width and height of bbox is 224, 224
		'''
		N = pred_tensor.size()[0]	# batchsize

		containobj_mask = target_tensor[:, :, :, 4] > 0	# in target_tensor, 2 bbox of 1 grid are the same, see yolodata
		noobj_mask = target_tensor[:, :, :, 4] == 0	# containobj_mask.size() == noobj_mask.size() == (N, 7, 7) 0-1 tensor
		containobj_mask = containobj_mask.unsqueeze(-1).expand_as(target_tensor)
		noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target_tensor)	# containobj_mask.size() == noobj_mask.size() 
		# (N, 7, 7) -> (N, 7, 7, 1) -> (N, 7, 7, 30)

		pred_containobj = pred_tensor[containobj_mask].view(-1, 30)	# pred_containobj.size() == (N_pred_containobj, 30), every grid in pred_containobj should have obj
		box_pred_containobj = pred_containobj[:, :10].contiguous().view(-1, 5)	# box_pred_containobj.size() == (2*N_pred_containobj, 5)
		class_pred_containobj = pred_containobj[:, 10:]	# class_pred_containobj.size() == (N_pred_containobj, 20)
		# without contiguous(), you will get a RunTime Error

		target_containobj = target_tensor[containobj_mask].view(-1, 30)	# target_containobj.size() == pred_containobj.size(), every grid in taregt_containobj has obj
		box_target_containobj = target_containobj[:, :10].contiguous().view(-1, 5)	# box_target_containobj.size() == box_pred_containobj.size()
		class_target_containobj = target_containobj[:, 10:]	# class_target_containobj.size() == class_pred_containobj.size()
		# without contiguous(), you will get a RunTime Error

	'''
	compute not contain obj loss
	'''
		pred_noobj = pred_tensor[noobj_mask].view(-1, 30)	# pred_noobj.size() = (N_pred_noobj, 30), every grid should have no obj
		target_noobj = target_tensor[noobj_mask].view(-1, 30)	# target_noobj.size() == pred_noobj.size(), every grid has no obj
		noobj_mask_1 = torch.ByteTensor(pred_noobj.size())	# a new mask to select conficdence only
		# noobj_mask_1.size() == pred_noobj.size(), dtype = torch.uint8(ByteTensor)
		noobj_mask_1.zero_()
		noobj_mask_1[:, 4] = 1
		noobj_mask_1[:, 9] = 1	# confidence of bbox1 and bbox2
		confidence_pred_noobj = pred_noobj[noobj_mask_1]	# confidence_pred_noobj.size() = (2*N_pred_noobj,)
		confidence_target_noobj = target_noobj[noobj_mask_1]	# confidence_target_noobj.size() = confidence_pred_noobj.size()
		
		# 2. no obj loss
		noobj_loss = F.mse_loss(
								confidence_pred_noobj, 
								confidence_target_noobj, 
								size_average = False)

	'''
	compute contain obj loss
	'''
		containobj_responsebbox_mask = torch.ByteTensor(box_target_containobj.size())	#	containobj_responsebbox_mask.size() == (2*N_pred_containobj, 5)
		containobj_responsebbox_mask.zero_()	# mask for bbox responsible for object
		containobj_irresponsebbox_mask = torch.ByteTensor(box_target_containobj.size())	#	containobj_responsebbox_mask.size() == (2*N_pred_containobj, 5)
		containobj_irresponsebbox_mask.zero_()	# mask for bbox not responsible for object

		boxIOU_gred_target = torch.zeros(box_target_containobj.size())	# boxIOU_gred_target.size() == (2*N_pred_containobj, 5)
		for i in range(0, box_target_containobj.size()[0], 2):	# for every grid that should contain object
			box1_pred_containobj = box_pred_containobj[i, i+2]	# box1_pred_containobj.size() = (2, 5)
			box1_pcon_xyxy = Variable(torch.FloatTensor(box1_pred_containobj.size()))	# [x, y, w, h] -> [x1, y1, x2, y2]
			box1_pcon_xyxy[:, :2] = box1_pred_containobj[:, :2] - 0.5*self.S*box1_pred_containobj[:, 2:4]
			box1_pcon_xyxy[:, 2:4] = box1_pred_containobj[:, :2] + 0.5*self.S*box1_pred_containobj[:, 2:4]
			# box1_pred_containobj[:, :2] == [x, y] relevant to grid
			# box1_pred_containobj[:, 2:4] == [w, h] relevant to image -> self.S*box1_pred_containobj[:, 2:4] relevant to grid
			box2_target_containobj = box_target_containobj[i].view(-1, 5)	# box2_taeget_containobj.size() = (1, 5), box_target_containobj[i] == box_target_containobj[i+1]
			box2_tcon_xyxy = Variable(torch.FloatTensor(box2_target_containobj.size()))
			box2_tcon_xyxy[:, :2] = box2_target_containobj[:, :2] - 0.5*self.S*box2_target_containobj[:, 2:4]
			box2_tcon_xyxy[:, 2:4] = box2_target_containobj[:, :2] + 0.5*self.S*box2_target_containobj[:, 2:4]

			iou = self.compute_iou(box1_pcon_xyxy[:, :4], box2_tcon_xyxy[:, :4])	# iou.size() == (2, 1)
			max_iou, max_index = iou.max(0)	# iou.max() only returns value, iou.max(0) returns value, index
			
			containobj_responsebbox_mask[i + max_index] = 1
			containobj_irresponsebbox_mask[i + 1 - max_index] = 1

			boxIOU_gred_target[i + max_index, 4] = max_iou	# train the confidence to equal iou

		# 1. coordinates loss & response loss
		responsebox_pred_containobj = box_pred_containobj[containobj_responsebbox_mask].view(-1, 5)	# responsebox_pred_containobj.size() == (N_pred_containobj, 5)
		responsebox_target_containobj = box_target_containobj[containobj_responsebbox_mask].view(-1, 5)	
		# responsebox_target_containobj.size() == responsebox_pred_containobj.size()
		responseboxIOU_gred_target = boxIOU_gred_target[containobj_responsebbox_mask].view(-1, 5)
		# responseboxIOU_gred_target.size() == responsebox_pred_containobj.size()

		coordinates_loss = F.mse_loss(
										responsebox_pred_containobj[:, :2], 
										responsebox_target_containobj[:, :2], 
										size_average = False)
							+ F.mse_loss(
										torch.sqrt(responsebox_pred_containobj[:, 2:4]),
										torch.sqrt(responsebox_target_containobj[:, 2:4]),
										size_average = False)


		response_loss = F.mse_loss(
										responsebox_pred_containobj[:, 4],
										responseboxIOU_gred_target[;, 4],
										size_average = False)

		# 3. irresponse loss
		irresponsebox_pred_containobj = box_pred_containobj[containobj_irresponsebbox_mask].view(-1, 5)	# irresponsebox_pred_containobj.size() == (N_pred_containobj, 5)
		irresponsebox_target_containobj = box_target_containobj[containobj_irresponsebbox_mask].view(-1, 5)	
		irresponsebox_target_containobj[:, 4] = 0

		irresponse_loss = F.mse_loss(
										irresponsebox_pred_containobj[:, 4], 
										irresponsebox_target_containobj[:, 4],
										size_average = False)

		# 4. class loss
		class_loss = F.mse_loss(
								class_pred_containobj, 
								class_target_containobj,
								size_average = False)

		return (
				self.l_coord*coordinates_loss
			  + response_loss
			  + self.l_noobj*noobj_loss
			  + class_loss
			  + response_loss + irresponse_loss)/N