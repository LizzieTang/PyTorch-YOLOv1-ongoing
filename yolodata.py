# -*- coding: utf-8 -*-
# load img data
# .txt >> img_name.jpg x1 y1 x2 y2 c x1 y1 x2 y2 c 本例图片中有两个目标

__author = 'Lizzie'


import os

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import cv2


'''
the blog I regered to modifies the input images with opencv,
it's said that modifications can somehow increase data amount,
technically these modifications can be skipped,
yet, to ensure the efficiency, I keep them,
as this version is written when my  opencv is on its way home, the result can't be ensured
'''



class YoloData(data.Dataset):
	'''
	class YoloDta, object i.e: trainset = YoloData(f_dir = FILE_DIR, 
												   fname_list = FILE_NAMES,
												   tmodify = True,
												   transform = [transforms.ToTensor()])
	args:
		f_dir >> YOLO data files' path, (str), i.e './data/VOCdevkit/VOC2012/allimgs/'
		fname_list >> YOLO data files' names, (str or list), i.e1 'voc12_trainval.txt',
															 i.e2 ['voc12_trainval.txt', 'voc07_trainval.txt']
		modify >> do modifications to data or not, (bool), i.e True
		transform >> transform YOLO data(like images) to tensors, >> (list), i.e [transforms.ToTensor()]
	'''
    def __init__(self, f_dir, fname_list, modify, transform):

        print('YoLoData initialize')
        self.IN_IMG = 448	# input of Net in yolonet.py must be 448*448*3
        self.DIR = FILE_DIR	# read files in this DIR so as to generate img,target for Dataset
        self.MODIFY = modify
        self.transform = transform
        self.img_names = []
        self.boxes = []
        self.labels = []
        self.MEAN = (123,117,104) # opencv default color space: BGR

        if isinstance(fname_list, list):
        	'''
            concatenate multiple files (in the fname_list) together.
            This is especially useful for voc07/voc12 combination.
            '''
            NEW_FNAME = 'newfname.txt'
            os.system('cat %s > %s' % (' '.join(fname_list), NEW_FNAME))
            fname_list = NEW_FNAME
        self.FNAME = fname_list	# data txt

        with open(self.PATH, 'r') as f:
            lines  = f.readlines()	

        for line in lines:	# each line stands for one image with all its objects info
            one_img_list = line.strip().split()	# (list), one img with its objects, i.e ['img_name.jpg', 'x', 'y', 'w', 'h', 'c']
            self.img_names.append(one_img_list[0])
            num_boxes = (len(one_img_list) - 1) // 5
            box=[]
            label=[]
            for i in range(num_boxes):
                x1 = float(one_img_list[1+5*i])		# x1, y1 upper left coordinates
                y1 = float(one_img_list[2+5*i])
                x2 = float(one_img_list[3+5*i])		# x2, y2 lower right coordinates
                y2 = float(one_img_list[4+5*i])
                c = one_img_list[5+5*i]		# class
                box.append([x1,y1,x2,y2])
                label.append(int(c)+1)
            self.boxes.append(torch.Tensor(box))	# (list), boxes[i].type = tensor, i.e [tensor([[x1,y1,x2,y2]]), tensor(...)] tensor.size([1,4])
            self.labels.append(torch.LongTensor(label))	# i.e [tensor([3]), tensor([9, 9, 20])] tensor.size([1])
        self.num_samples = len(self.boxes)	# num_samples >> howmany images


    def __getitem__(self,idx):
        imgname = self.img_names[idx]	# pick one image, use opencv to read image, 
        img = cv2.imread(os.path.join(self.DIR, imgname))
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()

        if self.modify:
        	'''
        	modify the image: flip, scale, blur, H,S,V, shift, crop
        	'''
            img, boxes = self.random_flip(img, boxes)
            img, boxes = self.randomScale(img, boxes)
            img = self.randomBlur(img)
            img = self.RandomBrightness(img)
            img = self.RandomHue(img)
            img = self.RandomSaturation(img)
            img,boxes,labels = self.randomShift(img, boxes, labels)
            img,boxes,labels = self.randomCrop(img, boxes, labels)

        h,w,_ = img.shape	# (height, width, channels)
        boxes /= torch.Tensor([w,h,w,h]).expand_as(boxes)	# boxes = tensor([x1/w, y1/h, x2/w, y2/h], [...])
        img = self.BGR2RGB(img) # because pytorch pretrained model use RGB
        img = self.subMean(img,self.MEAN) # 减去均值 figuring out what for
        img = cv2.resize(img,(self.IN_IMG, self.IN_IMG))	# resize to (448, 448, 3)
        target = self.encoder(boxes,labels)	# generate targets in YOLO format 7x7x30
        for t in self.transform:	# tansform cv(img) to tensor(img)
            img = t(img)

        return img,target


    def __len__(self):
        return self.num_samples


    def encoder(self,boxes,labels):	# each grid >> 2 bbox , 30 >> (xc_, yc_, w, h, confidence, xc_, yc_, w, h, confidence, class1, ..., class20)
        '''
        boxes >> tensor([[x1, y1, x2, y2],
        				 [..............]]) 
        labels >> tensor([c1, c2]) 
        return 7x7x30
        '''
        grid_num = 7
        grid_size = 1.0 / grid_num
        target = torch.zeros((grid_num, grid_num, 30))	# (7,7,30)
        wh = boxes[:, 2:] - boxes[:, :2]	# wh >> tensor([x2-x1, y2-y1], [...])
        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2	# cxcy >> tensor([xcenter, ycenter], [...])
        for i in range(cxcy.size()[0]):	# cxcy.size() >> (object_num, 2), for each object
            cxcy_sample = cxcy[i]	# cxcy_sample >> tensor([xcenter, ycenter])
            ij = (cxcy_sample*grid_num).floor() # boxcenter/imgsize -> grid_idx/7
            target[int(ij[1]),int(ij[0]),4] = 1	# confidence1 = 1
            target[int(ij[1]),int(ij[0]),9] = 1	# confidence2 = 1
            target[int(ij[1]),int(ij[0]),int(labels[i])+9] = 1
            x1y1 = ij * grid_size #匹配到的网格的左上角相对坐标
            delta_xy = (cxcy_sample - xy) / grid_size	# delta_xy >> tensor([xc_, yc_], [...]), 0< xc_ <1, xc_==(xcenter-x1)/grid_size
            target[int(ij[1]),int(ij[0]),2:4] = wh[i]
            target[int(ij[1]),int(ij[0]),:2] = delta_xy
            target[int(ij[1]),int(ij[0]),7:9] = wh[i]
            target[int(ij[1]),int(ij[0]),5:7] = delta_xy
        return target


    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    def BGR2HSV(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    def HSV2BGR(self,img):
        return cv2.cvtColor(img,cv2.COLOR_HSV2BGR)

    
    def RandomBrightness(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])	# pick num between 0.5, 1.5
            v = v*adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)	# x>255 >> x=255; x<0 >> x=0; 0<=x<=255 >> x不变
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr
    def RandomSaturation(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            s = s*adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr
    def RandomHue(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            h = h*adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr


    def randomBlur(self,bgr):
        if random.random()<0.5:
            bgr = cv2.blur(bgr,(5,5))
        return bgr


    def randomShift(self,bgr,boxes,labels):
        #平移变换
        center = (boxes[:,2:]+boxes[:,:2])/2
        if random.random() <0.5:
            height,width,c = bgr.shape
            after_shfit_image = np.zeros((height,width,c),dtype=bgr.dtype)
            after_shfit_image[:,:,:] = (104,117,123) #bgr
            shift_x = random.uniform(-width*0.2,width*0.2)
            shift_y = random.uniform(-height*0.2,height*0.2)
            #print(bgr.shape,shift_x,shift_y)
            #原图像的平移
            if shift_x>=0 and shift_y>=0:
                after_shfit_image[int(shift_y):,int(shift_x):,:] = bgr[:height-int(shift_y),:width-int(shift_x),:]
            elif shift_x>=0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),int(shift_x):,:] = bgr[-int(shift_y):,:width-int(shift_x),:]
            elif shift_x <0 and shift_y >=0:
                after_shfit_image[int(shift_y):,:width+int(shift_x),:] = bgr[:height-int(shift_y),-int(shift_x):,:]
            elif shift_x<0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),:width+int(shift_x),:] = bgr[-int(shift_y):,-int(shift_x):,:]

            shift_xy = torch.FloatTensor([[int(shift_x),int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:,0] >0) & (center[:,0] < width)
            mask2 = (center[:,1] >0) & (center[:,1] < height)
            mask = (mask1 & mask2).view(-1,1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if len(boxes_in) == 0:
                return bgr,boxes,labels
            box_shift = torch.FloatTensor([[int(shift_x),int(shift_y),int(shift_x),int(shift_y)]]).expand_as(boxes_in)
            boxes_in = boxes_in+box_shift
            labels_in = labels[mask.view(-1)]
            return after_shfit_image,boxes_in,labels_in
        return bgr,boxes,labels

    def randomScale(self,bgr,boxes):
        #固定住高度，以0.8-1.2伸缩宽度，做图像形变
        if random.random() < 0.5:
            scale = random.uniform(0.8,1.2)
            height,width,c = bgr.shape
            bgr = cv2.resize(bgr,(int(width*scale),height))
            scale_tensor = torch.FloatTensor([[scale,1,scale,1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return bgr,boxes
        return bgr,boxes

    def randomCrop(self,bgr,boxes,labels):
        if random.random() < 0.5:
            center = (boxes[:,2:]+boxes[:,:2])/2
            height,width,c = bgr.shape
            h = random.uniform(0.6*height,height)
            w = random.uniform(0.6*width,width)
            x = random.uniform(0,width-w)
            y = random.uniform(0,height-h)
            x,y,h,w = int(x),int(y),int(h),int(w)

            center = center - torch.FloatTensor([[x,y]]).expand_as(center)
            mask1 = (center[:,0]>0) & (center[:,0]<w)
            mask2 = (center[:,1]>0) & (center[:,1]<h)
            mask = (mask1 & mask2).view(-1,1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if(len(boxes_in)==0):
                return bgr,boxes,labels
            box_shift = torch.FloatTensor([[x,y,x,y]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            boxes_in[:,0]=boxes_in[:,0].clamp_(min=0,max=w)
            boxes_in[:,2]=boxes_in[:,2].clamp_(min=0,max=w)
            boxes_in[:,1]=boxes_in[:,1].clamp_(min=0,max=h)
            boxes_in[:,3]=boxes_in[:,3].clamp_(min=0,max=h)

            labels_in = labels[mask.view(-1)]
            img_croped = bgr[y:y+h,x:x+w,:]
            return img_croped,boxes_in,labels_in
        return bgr,boxes,labels


    def subMean(self,bgr,mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h,w,_ = im.shape
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:,0] = xmin
            boxes[:,2] = xmax
            return im_lr, boxes
        return im, boxes
    def random_bright(self, im, delta=16):



def main():
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    FILE_DIR = './data/VOCdevkit/voc2012/allimgs/'
    train_dataset = yoloDataset(f_dir = FILE_DIR,
    							fname_list = 'voc12_trainval.txt',
    							modify = True,
    							transform = [transforms.ToTensor()] )
    train_loader = DataLoader(train_dataset,
    						  batch_size=1,
    						  shuffle=False,
    						  num_workers=0)
    train_iter = iter(train_loader)
    for i in range(100):
        img,target = next(train_iter)
        print(img,target)

MAIN = False
if __name__ == '__main__' & MAIN:
    main()










