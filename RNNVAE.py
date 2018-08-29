import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms

from torch.autograd import Variable

#from torchaudio import transforms
#from torchaudio import Datasets
import os
import sys
import shutil
from glob import glob
import datetime
import re
# Import comet_ml in the top of your file
from comet_ml import Experiment
import sklearn
from sklearn.preprocessing import LabelEncoder

import scipy
import librosa
import matplotlib.pyplot as plt


import numpy as np

###HYPERPARAMETERS
"""
Utilise this when writing new tests"""
#hyper_params = {"learning_rate": 0.5, "steps": 100000, "batch_size": 50}
#experiment.log_multiple_params(hyper_params)
batch_size = 48
n_categories = 7
n_hidden = 128

input_size = 50
num_epochs = 1000
learning_rate = 1e-6


hidden_size = 128
num_layers = 2
num_classes = n_categories

# Create an experiment
experiment  = Experiment(api_key="u29M66IEzt5k2y7v0DST3liIM")



def Datapath_type():
    import os
    if os.name == 'nt':
        DATAPATH = 'D:\Masters\Test/'
    else:
        DATAPATH = '/media/tim/Elements/Masters/Test/'
    
    return DATAPATH
DATAPATH = Datapath_type()



class audioDataset(Dataset):
    """
    Instance of Dataset with additional code to create npy file with MFCCs should fix later.
    """
    
    
    def __init__(self,  DATAPATH, 
                 train = True,
                 process = True,
                 processed_folder = 'Processed/Torched',
                 processed_file = 'whale.pt',
                 root = os.path.expanduser(DATAPATH),
                 labels = 'labels.txt',
                 audio_files= 'labels.txt',
                 file = 'test1'):
        self.DATAPATH = DATAPATH
        self.processed_folder = processed_folder
        self.processed_file = processed_file    
        self.process = process
        self.root = root
        self.labels = labels
        self.audio_files = audio_files
        self.file = file
        self.train = train
        
        #Code not needed once data has been npy'd
        if process == True:
            self.preprocessor()
        


        self.data, self.labels = torch.load(os.path.join(
                self.root, self.processed_folder, self.processed_file))
        self.len = len(self.data)
 
        
    def __getitem__(self, index):
        data, labels = self.data[index], self.labels[index]
        
        return data, labels
    
    
    
    def __len__(self):
        #Reads in dimension 1 of the 3-D x_data array(width of dataset, length of MFCC, MFCC amount)
       
        return len(self.data)
    

    def preprocessor(self):
                #https://github.com/pytorch/audio/blob/master/torchaudio/datasets/yesno.py       
    
            # process and save as torch files
        print('Processing...')
        self.processed_folder = 'Processed/Torched'
        self.processed_file = 'whale.pt'      
        self.root = os.path.expanduser(self.DATAPATH)
        tensors = []
        labels = []
        lengths = []

        audios = [x for x in os.listdir(self.DATAPATH + '/Processed/') if ".wav" in x]
        lb_make = LabelEncoder()
        for i, f in enumerate(audios):
            if f.endswith(".wav"):
                #print(f)
                full_path = os.path.join(self.DATAPATH + '/Processed/' , f)
                sig, sr = scipy.io.wavfile.read(full_path)
                tensors.append(sig)
                #lengths.append(sig.size(0))
                label = os.path.basename(f).split(".", 1)[0].split("_")
                
                labels.append(label[0])
                

                
        label = lb_make.fit_transform(labels)
        label_out = torch.tensor(label)
        
                
                        # sort sigs/labels: longest -> shortest
                #tensors, labels = zip(*[(b, c) for (a, b, c) in sorted(
                #zip(lengths, tensors, labels), key=lambda x: x[0], reverse=True)])
                        #self.max_len = tensors[0].size(0)
                
                #labels = torch.FloatTensor(labels)
        torch.save(
                        (tensors, label_out),
                        os.path.join(
                        self.root,
                        self.processed_folder,
                        self.processed_file))

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNN, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        
        
    def forward(self, x, hidden_dim):
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)) 
       
        # Initialize cell state
        if torch.cuda.is_available():
            c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            
            c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        
        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))  
        
        # print (out.size())
        # Decode hidden state of last time step
        output = self.fc(out[:, -1, :])  
        
        return output
    

def main():



    train_dataset = audioDataset(DATAPATH = DATAPATH, train=True,process = False, file = 'test1')

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)            

    input_dim = 20
hidden_dim = 69
layer_dim = 1  
output_dim = 7

model = RNN(input_dim, hidden_dim, layer_dim, output_dim)
criterion = nn.CrossEntropyLoss()

learning_rate = 0.1
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for i, (data, labels) in enumerate(train_loader):
        data=data.type(torch.FloatTensor)
        data = Variable(data)
        data = (data, batch_size, input_dim)
        #labels = labels.view(-1)
        labels = labels.type(torch.LongTensor) 
        labels = Variable(labels)
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
    
        # Forward pass to get output/logits
        # outputs.size() --> 100, 10
        outputs = model(data, n_hidden)
        
        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)
        
        # Getting gradients w.r.t. parameters
        loss.backward()
        
        # Updating parameters
        optimizer.step()