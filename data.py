import os
import torch
#import cPickle as pickle
import pickle as pickle
import numpy as np
import torchvision.transforms as transforms


def unpickle(fname):
    with open(fname,'rb') as fin:
        unpickled_dict = pickle.load(fin,encoding='bytes')
    return unpickled_dict

class dataset():
    def __init__(self, root=None, train=True):
        """ Read the train/test dataset from file """
        self.root = root
        self.train = train
        self.transform = transforms.ToTensor()
        
        if self.train:
            train_data_path = os.path.join(root, 'data_batch_1')
            data_dict = unpickle(train_data_path)
            train_data_np = data_dict[b'data']
            self.train_labels = (np.array(data_dict[b'labels'])).astype('int')
            
            #for i in range(2,6):
            #    train_data_path = os.path.join(root, 'data_batch_'+str(i))
            #    data_dict = unpickle(train_data_path)
            #    train_data_np = np.vstack((train_data_np,data_dict[b'data']))
            #    self.train_labels = np.hstack((self.train_labels,(np.array(data_dict[b'labels'])).astype('int') ))

            self.train_data = torch.from_numpy( ( train_data_np.reshape(10000,3,32,32) ).astype('float32')  )


        else:
            test_data_path = os.path.join(root, 'test_batch')
            data_dict = unpickle(test_data_path)
            test_data_np = data_dict[b'data'].reshape(10000,3,32,32)
            self.test_data = torch.from_numpy( test_data_np.astype('float32') )
            self.test_labels = (np.array(data_dict[b'labels'])).astype('int')


    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]


        return img, target
