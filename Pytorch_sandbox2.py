
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms

from torch.autograd import Variable

#from torchaudio import transforms
#from torchaudio import Datasets
import os
import sys
from glob import glob
import datetime
import re

import sklearn 
import scipy
import librosa
import keras
import matplotlib.pyplot as plt


import numpy as np





class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


class data_creator:

    def __init__(self,
                 DATAPATH,
                 train=True,
                 labels='labels.txt',
                 audio_files='labels.txt',
                 x_data=np.zeros(shape=(100,)),
                 y_data=np.zeros(shape=(100,)),
                 file='test1'):

        self.DATAPATH = DATAPATH
        self.labels = labels
        self.audio_files = audio_files
        self.file = file
        self.train = train
        self.x_data = x_data
        self.y_data = y_data
        # Code not needed once data has been npy'd
        self.get_train_test()

    def get_train_test(self, split_ratio=0.75, random_state=42):
        # Get available labels
        self.get_labels()
        X = np.load(self.DATAPATH + self.labels[0])
        print(X.shape)

        y = np.zeros(len(X))

        # Getting first arrays

        # Append all of the dataset into one single array, same goes for y
        for i, label in enumerate(self.labels[1:]):
            path = self.DATAPATH + label
            label = re.sub('\.npy$', '', path)
            x = np.load(label + '.npy')
            X = np.vstack((X, x))
            y1 = np.ones(len(x))
            vals = self.label_indices[i + 1]
            mm = vals * y1
            y = scipy.sparse.hstack((y, mm))
            # y=np.hstack((y,mm))

        ###### HOW CAN I CONVERT AN ARRAY INTO 2D

        # y1 = y1*self.label_indices[i+1]
        # y = np.vstack((y, y1))

        # print(y)
        # for file in files:
        #    if file.endswith('.npy'):
        #        print(file)
        #        X = np.load(current_path+file)
        #        y =

        y = y.T
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=(1 - split_ratio),
                                                                                    random_state=random_state)

        training_set = (X_train, y_train)
        print(training_set)
        test_set = (X_test, y_test)

        np.save((self.DATAPATH + 'testing' + '_train'), training_set)
        np.save((self.DATAPATH + 'testing' + '_test'), test_set)
        # print(y)
        return

    def get_labels(self):
        """
        get_labels: Reads in the whale audio files from the DATAPATH folder and
        collects all the audio files that have been processed. This checks for
        the .npy file extension but discludes the _train and _test appendings to the file
        Usage:
        self.labels - For collecting class labels for the classifier.
        self.audio_labels - For creating the MFCC npy files from the audio data.


        """
        included_extensions = '.npy'
        excluded_extensions = set(['_train.npy', '_test.npy'])
        excluded_extensions_audio = set({'.npy'})
        # allows exclusion of bad folders and modified files

        self.labels = [fn for fn in os.listdir(self.DATAPATH)
                       if any(fn.endswith(ext) for ext in included_extensions)]
        self.labels = [fn for fn in self.labels
                       if not any(fn.endswith(exclude) for exclude in excluded_extensions)]

        self.audio_files = os.listdir(self.DATAPATH)
        self.audio_files = [fn for fn in self.audio_files if
                            not any(fn.endswith(exclude) for exclude in excluded_extensions_audio)]

        self.label_indices = np.arange(0, len(self.labels))
        self.categorical_class = keras.utils.np_utils.to_categorical(self.label_indices)

    # Handy function to convert wav2mfcc
    def wav2mfcc(self, file_path, max_len=40):
        mfcc = []
        rate, _ = wavfile.read(file_path)

        wave, sr = librosa.load(file_path, mono=True, sr=rate)

        wavelength = len(wave)
        if wavelength < sr:
            print('pass')
            pass

        else:
            mfcc = librosa.feature.mfcc(wave, sr=sr)

            # If maximum length exceeds mfcc lengths then pad the remaining ones
            if (max_len > mfcc.shape[1]):
                pad_width = max_len - mfcc.shape[1]
                mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

            # Else cutoff the remaining parts
            else:
                mfcc = mfcc[:, :max_len]

        return mfcc

    def save_data_to_array(self, max_len=11):

        self.get_labels()
        array_stack = []

        for label in self.audio_labels:

            wavfiles = [self.DATAPATH + label + '/' + wavfile for wavfile in os.listdir(self.DATAPATH + '/' + label)]

            exclude = set(['Noise.npy', 'Noise_train.npy', 'Noise_test.npy'])
            # allows exclusion of bad folders and modified files
            [wavfiles.remove(d) for d in list(wavfiles) if d in exclude]

            for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
                mfcc = self.wav2mfcc(wavfile, max_len=max_len)
                if len(mfcc) > 0:
                    mfcc_vectors.append(mfcc)

            np.save(label + '.npy', mfcc_vectors)

        return mfcc_vectors

    def build_npy():

        """
        Uses save_data_to_array which in turn uses wav2mfcc
        """
        save_data_to_array(max_len=11)


class audioDataset(Dataset):
    """
    Instance of Dataset with additional code to create npy file with MFCCs should fix later.
    """

    def __init__(self, DATAPATH,
                 train=True,
                 labels='labels.txt',
                 audio_files='labels.txt',
                 x_data=np.zeros(shape=(100,)),
                 y_data=np.zeros(shape=(100,)),
                 file='test1'):
        self.DATAPATH = DATAPATH
        self.labels = labels
        self.audio_files = audio_files
        self.file = file
        self.train = train
        self.x_data = x_data
        self.y_data = y_data
        # Code not needed once data has been npy'd

        if self.train == True:
            xy = np.load(self.DATAPATH + 'testing' + '_train' + '.npy')

        else:
            xy = np.load(self.DATAPATH + 'testing' + '_test' + '.npy')

        self.len = xy.shape[0]
        xy_dense = scipy.sparse.csr_matrix.todense(xy)

        x_data = Variable(torch.from_numpy(xy_dense[0]).float())
        self.x_data = torch.from_numpy(xy_dense[0])
        self.y_data = torch.from_numpy(xy_dense[1])

    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # Reads in dimension 1 of the 3-D x_data array(width of dataset, length of MFCC, MFCC amount)

        return self.len




def main():

    datacreator = data_creator(DATAPATH = 'D:\Masters\Test/')
    datacreator.load_train_test()
    #Whale dataset
    train_dataset = audioDataset(DATAPATH = 'D:\Masters\Test/', train=True, file = 'test1')

    test_dataset = audioDataset(DATAPATH = 'D:\Masters\Test/', train=False, file = 'test1')


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    print(test_loader)


if __name__ == "__main__":
    main()
