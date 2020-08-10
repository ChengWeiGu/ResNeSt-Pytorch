# 參考網站: https://wulc.me/2017/11/18/%E5%88%86%E6%89%B9%E8%AE%AD%E7%BB%83%E8%BF%87%E5%A4%A7%E7%9A%84%E6%95%B0%E6%8D%AE%E9%9B%86/

import numpy as np
import random
from os.path import basename, join, dirname, isfile, isdir
from os import listdir, walk
import cv2
import pickle
import gc
import random


def load_data(dataset_src):
    with open(dataset_src, "rb") as the_file:
        return pickle.load(the_file)
        the_file.close()


def train_test_batch_generator(data_dir = r'D:\DataSet\PCB\data5/'):
    data_basename = listdir(data_dir)
    Train_basename = [name for name in data_basename if (name.startswith('Train'))]
    Test_basename = [name for name in data_basename if (name.startswith('Test')) ]
    # num_items = len(Train_basename)
    
    _trainX , _trainY = np.array([]), np.array([])
    _testX , _testY = np.array([]), np.array([])

    for train_file, test_file in zip(Train_basename, Test_basename):
        print('\n =====> Reading file {} and {} ...........\n'.format(train_file,test_file))
        # num_items -= 1
        # gc.collect()
        _trainX , _trainY = load_data(data_dir + train_file)
        _testX , _testY = load_data(data_dir + test_file)

        yield (_trainX , _trainY, _testX , _testY)
        _trainX , _trainY = np.array([]), np.array([])
        _testX , _testY = np.array([]), np.array([])
        print('\n =====>  file {} and {} finished...........\n'.format(train_file,test_file))
        
        
def train_batch_generator(data_dir = './data/'):
    data_basename = listdir(data_dir)
    Train_basename = [name for name in data_basename if (name.startswith('Train'))]
    
    _trainX , _trainY = np.array([]), np.array([])

    for train_file in Train_basename:
        print('\n =====> Reading training file {} ...........\n'.format(train_file))

        _trainX , _trainY = load_data(data_dir + train_file)

        yield (_trainX , _trainY)
        _trainX , _trainY = np.array([]), np.array([])
        print('\n =====>  training file {} finished...........\n'.format(train_file))
        
        
def test_batch_generator(data_dir = './data/'):
    data_basename = listdir(data_dir)
    Test_basename = [name for name in data_basename if (name.startswith('Test')) ]
    
    _testX , _testY = np.array([]), np.array([])

    for test_file in Test_basename:
        print('\n =====> Reading testing file {} ...........\n'.format(test_file))

        _testX , _testY = load_data(data_dir + test_file)

        yield (_testX , _testY)
        _testX , _testY = np.array([]), np.array([])
        print('\n =====>  file {} finished...........\n'.format(test_file))
            