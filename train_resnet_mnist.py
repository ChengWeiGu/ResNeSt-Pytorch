import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
# import seaborn as sns 
# import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import pickle
import time
import argparse


def change_output_layers(model):
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # change 3 chanels to 1 chanel for gray scale
    model.fc = nn.Linear(2048, classes, bias=True)
    return model


def load_pkl_data(dataset_dst):
    with open(dataset_dst, "rb") as the_file:
        return pickle.load(the_file)
        the_file.close()
        


def preprocess(mode ='train'):
    
    axes = ['x','y','z']
    pass_dir = r"D:\Side Work Data\PHM\Demo_box\demo_box_0518\wav\pass"
    ng1_dir = r"D:\Side Work Data\PHM\Demo_box\demo_box_0518\wav\ng1"
    ng2_dir = r"D:\Side Work Data\PHM\Demo_box\demo_box_0518\wav\ng2"
    n_points = 1600
    all_data = np.array([])
    all_label_data = np.array([])
    
    
    
    for ax in axes:
        
        pass_filename = ""
        ng1_filename = ""
        ng2_filename = ""
        
        if mode == 'train':
            pass_filename = pass_dir + '/' + ax + '/' + '20210518112208.csv'
            ng1_filename = ng1_dir + '/' + ax + '/' + '20210518114530.csv'
            ng2_filename = ng2_dir + '/' + ax + '/' + '20210518134847.csv'
        elif mode == 'test':
            pass_filename = pass_dir + '/' + ax + '/' + '20210518112338.csv'
            ng1_filename = ng1_dir + '/' + ax + '/' + '20210518114602.csv'
            ng2_filename = ng2_dir + '/' + ax + '/' + '20210518134922.csv'
        
        _temp_data = np.array([])
        _temp_lable_data = np.array([])
        
        for filename, label in zip([pass_filename, ng1_filename, ng2_filename],[0,1,2]):
            print('processing for file {}'.format(filename))
            
            data = pd.read_csv(filename).values[1600:59200] # 36 rows for n = 1600
            data = data.reshape((-1,n_points))
            n_samples = len(data)
            label_data = np.array([label]*n_samples).reshape((-1,1))
            
            if label == 0:
                _temp_data = data
                _temp_lable_data = label_data
            else:
                _temp_data = np.vstack((_temp_data,data))
                _temp_lable_data = np.vstack((_temp_lable_data,label_data))
                
        
        if ax == 'x':
            all_data = _temp_data
            all_label_data = _temp_lable_data
        else:
            all_data = np.hstack((all_data, _temp_data))
        
        
    
    # print(all_data.shape)
    # print()
    return np.hstack((all_data,all_label_data))
    
    

class PHM_dataset(Dataset):
    def __init__(self, all_data):
        
        self.rows = len(all_data)
        self.imgnp = all_data[:self.rows, :-1]
        self.labels = all_data[:self.rows, -1]
    
    def __len__(self):
        return self.rows
    
    def __getitem__(self, idx):
        image = torch.tensor(self.imgnp[idx], dtype=torch.float)
        image = image.view(3, 40, 40)  # (channel, height, width)
        label = torch.tensor(self.labels[idx], dtype = torch.long)
        return (image, label)



class MNIST_dataset(Dataset):
    def __init__(self, df, rows=42000):
        self.imgnp = df.iloc[:rows, 1:].values
        self.labels = df.iloc[:rows, 0].values
        self.rows = rows
    
    def __len__(self):
        return self.rows
    
    def __getitem__(self, idx):
        image = torch.tensor(self.imgnp[idx], dtype=torch.float) / 255  # Normalize
        image = image.view(1, 28, 28)  # (channel, height, width)
        label = self.labels[idx]
        return (image, label)




if __name__ == "__main__":
    
    parser=argparse.ArgumentParser()
    help_="train the model"
    parser.add_argument("-t","--train",help=help_,action="store_true")
    help_="verify the model"
    parser.add_argument("-v","--verify",help=help_,action="store_true")
    help_="load the previous weights"
    parser.add_argument("-w","--weights",help=help_)
    args=parser.parse_args()
    
    lr = 0.0001
    batch_size = 6
    epochs = 10
    classes = 10
    display_frequency = 6
    
    model = models.resnet50(pretrained=True)
    model = change_output_layers(model)
    print(model)

    if(torch.cuda.is_available()):
        model = model.cuda()
        torch.backends.cudnn.benchmark=True
        
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    
    train_df = pd.read_csv('train.csv')
    train_dataset = MNIST_dataset(train_df, 300)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    test_df = pd.read_csv('test.csv')
    test_dataset = MNIST_dataset(train_df, 50)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    
    # _train_data = preprocess('train')
    # train_dataset = PHM_dataset(_train_data)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # _test_data = preprocess('test')
    # test_dataset = PHM_dataset(_test_data)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    
    if args.train:
        
        last_loss = float('Inf')
        current_loss = 0                                      
                                                
        print("len of train_loader = %d" %len(train_loader))
        print("len of test_loader = %d" %len(test_loader))      

        print('Start Training....')
        train_losses, val_losses = [], []
        train_accu, val_accu = [], []
        
        start = time.time()
        
        if args.weights:
            model.load_state_dict(torch.load(args.weights))
        
        for epoch in range(epochs):
            epoch_start_time=time.time()
            running_loss=0
            accuracy=0
            sub_total = 0
            display_loss=0
            for i, data in enumerate(train_loader, 1):
                images, labels = data
                
                if(torch.cuda.is_available()):
                    images = images.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad()    
                outputs = model(images)

                _, predicted = torch.max(outputs.data, 1)
                sub_total += labels.size(0)
                accuracy += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                running_loss+=loss.item()
                display_loss+=loss.item()
                if(i%display_frequency == (display_frequency-1)):
                    print('Epoch: {} Batch: {}/{} loss: {:.6f} acc:{:.2f}'.format(epoch+1, i, len(train_loader), display_loss/display_frequency,(accuracy/sub_total)*100))
                    display_loss = 0

                loss.backward()
                optimizer.step()
            
            # saving best model according to loss
            current_loss = running_loss/len(train_loader)
            if current_loss < last_loss:
                print("loss improved from {:.6f} to {:.6f}...saving best model".format(last_loss,current_loss))
                torch.save(model.state_dict(), 'best_weights.pkl')
                last_loss = current_loss
            else:
                print("loss did not improve from {:.6f} to {:.6f}...".format(last_loss,current_loss))
                
            
            
            train_losses.append(current_loss)
            train_accu.append((accuracy/sub_total)*100)
            
            val_loss=0
            accuracy=0
            sub_total = 0
            with torch.no_grad():
                for j, data in enumerate(test_loader,1):
                    images, labels = data

                    if(torch.cuda.is_available()):
                        images = images.cuda()
                        labels = labels.cuda()
 
                    outputs = model(images)

                    _, predicted = torch.max(outputs.data, 1)
                    sub_total += labels.size(0)
                    accuracy += (predicted == labels).sum().item()
                    val_loss += criterion(outputs, labels).item()
            
            
            val_losses.append(val_loss/len(test_loader))
            val_accu.append((accuracy/sub_total)*100)
            
            
            print("Summary at Epoch: {}/{}..".format(epoch+1,epochs),
            "Time: {:.2f}s..".format(time.time()-epoch_start_time),
            "Training_Loss: {:.3f}..".format(train_losses[-1]),
            "Training_Accu: {:.3f}%..".format(train_accu[-1]),
            "Val_Loss: {:.3f}..".format(val_losses[-1]),
            "Val_Accu: {:.3f}%".format(val_accu[-1]))
            
        
        torch.save(model.state_dict(), 'resnet50_last_weights.pkl')
        # torch.save(model, 'resnet50.pkl')
        print('Training Completed in: {} secs'.format(time.time()-start))
    
    
    if args.weights and args.verify:
        
        model.load_state_dict(torch.load(args.weights))
                                                   
                                                
        print("len of train_loader = %d" %len(train_loader))
        print("len of test_loader = %d" %len(test_loader)) 
        
        #------------------------confusion matrix--------------------------#
        print('Calculate confusion matrix of testing data....')
        start = time.time()
        
        predictions = torch.LongTensor()
        for i, data in enumerate(test_loader, 1):
            images, labels = data
            images = images.cuda()
            
            outputs = model(images)
            
            pred = outputs.cpu().data.max(1, keepdim=True)[1]
            predictions = torch.cat((predictions, pred), dim=0)
            
        print('Test Completed in {} secs'.format(time.time() - start))
        test_matrix = confusion_matrix(test_dataset.labels , predictions.numpy())
        
        
        print('Calculate confusion matrix of training data....')
        start = time.time()
        
        predictions = torch.LongTensor()
        for i, data in enumerate(train_loader, 1):
            images, labels = data
            images = images.cuda()

            outputs = model(images)
            
            pred = outputs.cpu().data.max(1, keepdim=True)[1]
            predictions = torch.cat((predictions, pred), dim=0)
            
        print('Train Completed in {} secs'.format(time.time() - start))
        train_matrix = confusion_matrix(train_dataset.labels , predictions.numpy())
        
        
        for matrix , filename in zip([test_matrix,train_matrix],["test_matrix.csv", "train_matrix.csv"]):
            df_mat = pd.DataFrame(matrix)
            df_mat.to_csv(filename)
            
        print("test matrix:\n",test_matrix)
        print("train matrix:\n",train_matrix)
        
        #------------------------confusion matrix--------------------------#