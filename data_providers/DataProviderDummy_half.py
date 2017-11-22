# from commit e57a711, https://github.com/AllenCellModeling/pytorch_segmentation_classifier

import glob
import os
import numpy as np
from scipy import misc
# from natsort import natsorted
from PIL import Image
import torch
# import h5py
# import pandas as pd
import copy
import random
# from tqdm import tqdm

import hashlib

import pdb

# import aicsimage.processing as proc
# from aicsimage.io import tifReader

from data_providers.DataProviderABC import DataProviderABC 

class DataProvider(DataProviderABC):
    
    def __init__(self, image_parent, csv_name='data_jobs_out.csv', opts={}, imsize = [0, 128, 96, 64]):
        self.data = {}
        
        opts_default = {'rotate': False,
                        'hold_out': 1/10,
                        'verbose': True,
                        'target_col':'structureProteinName',
                        'channelInds': [0, 1, 2],
                        'h5_file': True,
                        'check_files':True,
                        'split_seed': 1}
                
        # set default values if they are missing
        for key in opts_default.keys(): 
            if key not in opts: 
                opts[key] = opts.get(key, opts_default[key])
        
        self.opts = opts
        
        self.imsize = imsize   
        self.nclasses = 10
        self.ndat = 5000
        
        self.halfpool = torch.nn.AvgPool3d(3, 2, 1)
        
    def get_n_dat(self, train_or_test = 'train'):
        return self.ndat
    
    def get_n_classes(self):
        return self.nclasses
        
    def get_image_paths(self, inds_tt, train_or_test):
        
        return None
        
    def get_images(self, inds_tt, train_or_test):
        dims = list(self.imsize)
        
        dims[0] = len(self.opts['channelInds'])
        dims.insert(0, len(inds_tt))
        
        
        image = np.random.normal(0, 1, size = dims)
        
        images = torch.from_numpy(image).float()
        
        images = self.halfpool(torch.autograd.Variable(images, volatile=True)).data
    
        return images
    
    def get_classes(self, inds_tt, train_or_test, index_or_onehot = 'index'):
        inds_master = self.data[train_or_test]['inds'][inds_tt]

        
        labels_tmp = np.random.choice(self.nclasses, len(inds_tt))

        if index_or_onehot == 'index':
            labels = labels_tmp
        else:
            labels = np.zeros([len(inds_tt), self.get_n_classes()])
            for c,i in enumerate(labels_tmp):
                labels[c,:] = i
            
            labels = torch.from_numpy(labels).long()
            
        labels = torch.LongTensor(labels)
        return labels
    
#     # train minibatches cycle through entire data set
#     # test minibatches are random draws each time to match the size of train minibatch
#     def make_random_minibatch_inds_train_and_test(self, batch_size=8):
        
#         n_train = self.get_n_dat('train')
#         inds_train_shuf = random.sample(range(n_train),n_train)
#         mini_batches_inds_train = [inds_train_shuf[i:i + batch_size] for i in range(0, len(inds_train_shuf), batch_size)]
        
#         mini_batch_train_lens = [len(b) for b in mini_batches_inds_train]
#         n_test = self.get_n_dat('test')
#         mini_batches_inds_test = [random.sample(range(n_test),b_size) for b_size in mini_batch_train_lens]
        
#         minibatch_inds_list = {}
#         minibatch_inds_list['train'] = mini_batches_inds_train
#         minibatch_inds_list['test'] = mini_batches_inds_test
        
#         return(minibatch_inds_list)
