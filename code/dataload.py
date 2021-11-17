import os
import numpy as np

import torch
import torch.nn as nn
from PIL import Image


## 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None,target_transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

        lst_data = os.listdir(self.data_dir)

        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):

        label = np.asarray(Image.open(os.path.join(self.data_dir, self.lst_label[index])))
        input = np.asarray(Image.open(os.path.join(self.data_dir, self.lst_input[index])))

        #label = np.round(label/255)*255


        if label.ndim == 2:
            label = label[:, :,np.newaxis ]
            #print(np.max(label))
        if input.ndim == 2:
            input = input[:, :,np.newaxis]
            #print(np.max(input))
        #data = np.array([])
        #data += [[input]]
        #data += [[label]]


        data = {'Input_ID': self.lst_input[index], 'Label_ID': self.lst_label[index], 'input': input, 'label': label}
        if self.transform:
            #print(input.shape)
            arr = np.concatenate([input, label], axis=2)
            #print(arr.shape)
            arr_t = self.transform(arr)
            #print(arr_t[:][:][0].shape)


            import matplotlib.pyplot as plt


            inputs = arr_t[:][:][0]
            #plt.imshow(inputs )
            #plt.show()
            inputs = inputs.reshape(1,inputs.shape[0],inputs.shape[1])
            labels = arr_t[:][:][1]

            #plt.imshow(labels)
            #plt.show()

            labels  = labels.reshape(1,labels.shape[0], labels.shape[1])


            data_t = {'Input_ID': self.lst_input[index],'Label_ID': self.lst_label[index],'input' : inputs,'label' : labels }
            data =[]
            data = data_t





        return data


## 트렌스폼 구현하기
class ToTensor(object):
    def __call__(self, data):
        input =  data
        input = input.reshape(input.shape[0],input.shape[1],2)
        input = input.transpose((2, 0, 1)).astype(np.float32)
        data =input

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        input = np.asarray(data)


        inputs = (input )/255

        data = inputs

        return data
