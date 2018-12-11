
# coding: utf-8

# In[1]:


from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.image as mpimg
# get_ipython().run_line_magic('matplotlib', 'inline')


annotation = loadmat("./clothing-co-parsing/annotations/pixel-level/0001.mat")['groundtruth']
image = mpimg.imread('./clothing-co-parsing/photos/0001.jpg')
imshow(image)
plt.show()
plt.imshow(annotation)
print(annotation[100, 250:300])
label_list = loadmat('./clothing-co-parsing/label_list.mat')
print(label_list['label_list'][0][19])
print(label_list['label_list'][0][41])
print(image.shape)


# print(label_list['label_list'][0][55])
print(label_list['label_list'][0])


# In[20]:


# define all pixel having a label will be marked as 1, other than that 0. 

import sys
sys.path.append("/anaconda2/envs/deep-learning/bin/python3")

import os
import numpy as np
from pathlib import Path
from skimage import io
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch as t
from torch.utils import data
from torchvision import transforms as tsf
import cv2

TRAIN_PATH = './train.pth'
TEST_PATH = './test.pth'
ANNOTATION_PATH = './clothing-co-parsing/annotations/pixel-level/'
MODEL_PATH = './checkpoints/'
N_EPOCH = 250
N_train_start = 0 
N_train_end = 1004
N_test_start = 1005
N_test_end = 1015

""" network input size. """
image_rows = 96
image_cols = 96

def process(file_path, has_mask, start, end):
    file_path = Path(file_path)
    files = sorted(list(Path(file_path).iterdir()))
    datas = []
    for file in tqdm(files[start:end]): # only process N_train images.
        item = {}
        imgs = []
        img = cv2.imread(str(file)) # has 3 channels.   
        img = cv2.resize(img, dsize=(image_rows, image_cols), interpolation=cv2.INTER_CUBIC)
        file_id = str(file).split('/')[-1].split('.')[0]
        if has_mask:
            mask_file = os.path.join(ANNOTATION_PATH, file_id + '.mat')
            this_mask = loadmat(mask_file)['groundtruth']
            this_mask[this_mask > 0] = 1 # set all annotation be same label.
            this_mask = cv2.resize(this_mask, dsize=(image_rows, image_cols), interpolation=cv2.INTER_CUBIC)
            item['mask'] = t.from_numpy(this_mask)
        item['name'] = file_id
        item['img'] = t.from_numpy(img)
        datas.append(item)
    return datas

if not os.path.exists(TRAIN_PATH):
    train_data = process('./clothing-co-parsing/photos/', True, N_train_start, N_train_end)
    t.save(train_data, TRAIN_PATH)
else:
    train_data = t.load(TRAIN_PATH)

if not os.path.exists(TEST_PATH):
    test_data = process('./clothing-co-parsing/photos/', False, N_test_start, N_test_end)
    t.save(test_data, TEST_PATH)
else:
    test_data = t.load(TEST_PATH)


# In[15]:


import PIL
class Dataset():
    def __init__(self,data,source_transform,target_transform):
        self.datas = data
#         self.datas = train_data
        self.s_transform = source_transform
        self.t_transform = target_transform
    def __getitem__(self, index):
        data = self.datas[index]
        img = data['img'].numpy()
        mask = data['mask'][:,:,None].byte().numpy()
        img = self.s_transform(img)
        mask = self.t_transform(mask)
        return img, mask
    def __len__(self):
        return len(self.datas)
s_trans = tsf.Compose([
    tsf.ToPILImage(),
    tsf.Resize((128,128)),
    tsf.ToTensor(),
    tsf.Normalize(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5])
]
)
t_trans = tsf.Compose([
    tsf.ToPILImage(),
    tsf.Resize((128,128),interpolation=PIL.Image.NEAREST),
    tsf.ToTensor(),]
)
dataset = Dataset(train_data,s_trans,t_trans)
dataloader = t.utils.data.DataLoader(dataset,num_workers=2,batch_size=4)


# In[21]:


# img,mask = dataset[12]
# plt.subplot(121)
# plt.imshow(img.permute(1,2,0).numpy()*0.5+0.5)
# plt.subplot(122)
# plt.imshow(mask[0].numpy())
# plt.savefig('train.png')


# In[17]:

from torch import nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )
    
    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = t.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = t.nn.functional.sigmoid(x)
        return x


""" Load model """ 
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# def get_model():
#     model = unet11(pretrained=True)
#     model.eval()
#     return model.to(device)


def soft_dice_loss(inputs, targets):
    num = targets.size(0)
    m1  = inputs.view(num,-1)
    m2  = targets.view(num,-1)
    intersection = (m1 * m2)
    score = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
    score = 1 - score.sum()/num
    return score
        



def train():
    model = UNet(3,1).cuda() # use gpu
    optimizer = t.optim.Adam(model.parameters(),lr = 1e-3)

    for epoch in range(N_EPOCH):
        loss_total = 0
        for x_train, y_train  in tqdm(dataloader):
            x_train = t.autograd.Variable(x_train).cuda() # use gpu
            y_train = t.autograd.Variable(y_train).cuda() # use gpu
            optimizer.zero_grad()
            o = model(x_train)
            loss = soft_dice_loss(o, y_train)
            loss_total += loss
            loss.backward()
            optimizer.step()
        if epoch % 20 == 0:
            t.save(model.state_dict(), MODEL_PATH + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))
        print('epoch {} done. loss: {}'.format(epoch + 1, loss_total))


""" Save and load model. """
model_version = 21
base_model = MODEL_PATH + 'CP{}.pth'.format(model_version) # set model version here.

if not os.path.exists(base_model):
    train()
else:
    model = UNet(3,1)
    model.load_state_dict(t.load(base_model, map_location='cpu'))
    # model.load_state_dict(t.load(base_model))

        
class TestDataset():
    def __init__(self,path,source_transform):
        self.datas = t.load(path)
        self.s_transform = source_transform
    def __getitem__(self, index):
        data = self.datas[index]
        img = data['img'].numpy()
        img = self.s_transform(img)
        return img
    def __len__(self):
        return len(self.datas)

testset = TestDataset(TEST_PATH, s_trans)
testdataloader = t.utils.data.DataLoader(testset,num_workers=2,batch_size=2)

# import pydensecrf.densecrf as dcrf

# def dense_crf(img, output_probs):
#     h = output_probs.shape[0]
#     w = output_probs.shape[1]

#     output_probs = np.expand_dims(output_probs, 0)
#     output_probs = np.append(1 - output_probs, output_probs, axis=0)

#     d = dcrf.DenseCRF2D(w, h, 2)
#     U = -np.log(output_probs)
#     U = U.reshape((2, -1))
#     U = np.ascontiguousarray(U)
#     img = np.ascontiguousarray(img)

#     d.setUnaryEnergy(U)

#     d.addPairwiseGaussian(sxy=20, compat=3)
#     d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=img, compat=10)

#     Q = d.inference(5)
#     Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

#     return Q

model = model.eval()
for i, data in enumerate(testdataloader):
    data = t.autograd.Variable(data, volatile=True)#.cuda())
    o = model(data)
    new_mask = o[1][0].data.cpu().numpy()
    input_image = data[1].data.cpu().permute(1,2,0).numpy()*0.5+0.5
    # tm = dense_crf(np.array(input_image).astype(np.uint8), tm)
    # new_mask = tm > 0.5

    plt.subplot(121)
    plt.imshow(input_image)
    plt.subplot(122)
    plt.imshow(new_mask)
    plt.savefig('./results/CP{}_{}.png'.format(model_version, i))
    # break