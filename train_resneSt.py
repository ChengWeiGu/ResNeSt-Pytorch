import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
# import seaborn as sns 
# import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from resnest.torch import resnest50, resnest101
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import pickle
import time
import argparse


def change_output_layers(model):
    # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # change 3 chanels to 1 chanel for gray scale
    model.fc = nn.Linear(2048, classes, bias=True)
    return model


def load_pkl_data(dataset_dst):
    with open(dataset_dst, "rb") as the_file:
        return pickle.load(the_file)
        the_file.close()
        

class LCD_data(Dataset):
    """MNIST dtaa set"""
    
    def __init__(self, file_path, 
                 transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), 
                     transforms.Normalize(mean=(0.5,), std=(0.5,))])
                ):
        
        self.X, self.y = load_pkl_data(file_path)
        self.X = self.X.astype(np.uint8)
        self.y = torch.tensor(self.y, dtype = torch.long)
            
        self.transform = transform
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.transform(self.X[idx]), self.y[idx]



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
    batch_size = 32
    epochs = 50
    classes = 5
    display_frequency = 40
    dataset_train_dst = r'..\dataset_5cls\dataset_train_lcd.pkl'
    dataset_test_dst = r'..\dataset_5cls\dataset_test_lcd.pkl'
    
    model = resnest50(pretrained=True)    
    # model = resnest101(pretrained=True)                                
    # model = models.resnet50(pretrained=True)
    model = change_output_layers(model)
    # print(model)

    if(torch.cuda.is_available()):
        model = model.cuda()
        torch.backends.cudnn.benchmark=True
        
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    
    
    if args.train:
        
        last_loss = float('Inf')
        current_loss = 0

        train_dataset = LCD_data(dataset_train_dst, transform= transforms.Compose(
                                    [transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]))
        test_dataset = LCD_data(dataset_test_dst)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   batch_size=batch_size, shuffle=False)
                                                   
                                                
        print("len of train_loader = %d" %len(train_loader))
        print("len of test_loader = %d" %len(test_loader))                                      
        

        print('Start Training....')
        train_losses, val_losses = [], []
        train_accu, val_accu = [], []
        
        total = 0
        train_acc = 0
        start = time.time()
        
        if args.weights:
            model.load_state_dict(torch.load(args.weights))
        
        for epoch in range(epochs):
            epoch_start_time=time.time()
            running_loss=0
            accuracy=0
            sub_total = 0
            for i, data in enumerate(train_loader, 1):
                images, labels = data
                
                if(torch.cuda.is_available()):
                    images = images.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad()    
                outputs = model(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                sub_total += labels.size(0)
                train_acc += (predicted == labels).sum().item()
                accuracy += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                running_loss+=loss.item()
                if(i%display_frequency == 0):
                    print('Epoch: {} Batch: {}/{} loss: {:.6f} acc:{:.2f}'.format(epoch+1, i, len(train_loader), loss.item(),(accuracy/sub_total)*100))

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
                
            
            
            train_losses.append(running_loss/len(train_loader))
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
            
        
        torch.save(model.state_dict(), 'resnet50_weights.pkl')
        # torch.save(model, 'resnet50.pkl')
        print('Training Completed in: {} secs'.format(time.time()-start))
        print('Training accuracy: {} %'.format((train_acc/total)*100))
    
    
    if args.weights and args.verify:
        
        model.load_state_dict(torch.load(args.weights))
        train_dataset = LCD_data(dataset_train_dst, transform= transforms.Compose(
                                    [transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]))
        test_dataset = LCD_data(dataset_test_dst)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=24, shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   batch_size=24, shuffle=False)
                                                   
                                                
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
        test_matrix = confusion_matrix(test_dataset.y , predictions.numpy())
        
        
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
        train_matrix = confusion_matrix(train_dataset.y , predictions.numpy())
        
        
        for matrix , filename in zip([test_matrix,train_matrix],["test_matrix.csv", "train_matrix.csv"]):
            df_mat = pd.DataFrame(matrix)
            df_mat.to_csv(filename)
            
        print("test matrix:\n",test_matrix)
        print("train matrix:\n",train_matrix)
        
        #------------------------confusion matrix--------------------------#
        
    
    
    
    
