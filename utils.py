
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import sys
import os
from tqdm import tqdm
#DATA ANALYSIS TOOLS
import tensorflow as tf
import keras
import librosa
import numpy as np
import pandas as pd
import scipy
from scipy.io import wavfile
from sklearn.model_selection import train_test_split


#FILE IMPORTERS
import csv


#SYSTEM TOOLS
import os
import glob
import tkinter as tk
from tkinter import filedialog as filedialog
import fnmatch
from itertools import compress


invalid_training_type = "Error invalid training Type"

FLAGS = None





class logfile_reformatter:


    def __init__(self,
                 file_in = "/tmp", 
                 file_out = "/tmp", 
                 filepath = "/tmp",
                 name = "dummy.txt"):

        self.file_in = file_in
        self.file_out = file_out
        self.filepath = filepath
        self.name = name
        
    def get_file_in(self):
        return self.file_in

    def get_file_out(self):
        return self.file_out

    def get_filepath(self):
        return self.filepath

    def get_name(self):
        return self.name

    def set_file_in(self, vals):
        self.file_in = vals

    def set_file_out(self, vals):
        self.file_out = vals    

    def set_filepath(self, vals):
        self.filepath = vals

    def set_name(self, vals):
        self.name = vals

    def reformatter(self):
        
        data = pd.read_csv(self.file_in, delim_whitespace = True, header = None, error_bad_lines=False)
        data = data.dropna(axis = 1, how = 'any')
        df = data.iloc[:,[0, 1]]

        
        #data.drop([3,4,5,6], axis = 1)
        #data.columns = ["start time", "end time"]"/home/Documents/Masters/
        try:
            a = df[1].str.contains("start")
            if a.empty == False:
                a = df.drop(df.index[[0]])
                df = a
        except:
            pass
        vals = list(df.columns.values)
    
        
        
        df.columns = ["start time", "end time"]
        
            
        df.to_csv(self.file_out, sep = ' ',index = False)
        return df
    
    def output_file_creator(self):
        file_out = []
        file_out = self.filepath + '/' + self.name + '.log'
        self.set_file_out(file_out)

    def clean_log_files(self):
        locat = filedialog.askdirectory()
        
        for folders, dirs, files in os.walk(locat + "/"):
            
            for file in glob.glob(folders + "/*"):
                if file.endswith(".log") or file.endswith(".box"):
                    base = os.path.basename(file)
                    name, _ = os.path.splitext(base)
                    self.set_name(name)
                    (filepath, _) = os.path.split(file)
                    self.set_filepath(filepath)
                    self.output_file_creator()
                    self.set_file_in(file)
                    
                    df = self.reformatter()
    
class audio_builder:
    def __init__(self, 
        filename =  "/home/Documents/Masters/test4/dummy.wav",  
        output_file = "/home/Documents/Masters/test4/dummy_mod.wav",
        output_file_noise ="/home/Documents/Masters/test4/dummy_mod_noise.wav"): 

        self.filename = filename
        self.output_file = output_file
        self.output_file_noise = output_file_noise

    def set_filename(self, vals):
        self.filename = vals

    def get_filename(self):
        return self.filename

    def set_output_file(self, vals):
        self.output_file = vals

    def get_output_file(self):
        return self.output_file

    def set_output_file_noise(self, vals):
        self.output_file_noise = vals

    def get_output_file_noise(self):
        return self.output_file_noise



    def get_and_save_audio(self, start, end):
        """
        Loads in audio snippets according to the log files and outputs snippets wave files.
        
        """      

        rate, _ = wavfile.read(self.get_filename())
        elapsed = end - start
        elapsed60percent = elapsed*.60
    
        start_edit = start - elapsed60percent
        duration = elapsed + elapsed60percent       
        data, sr = librosa.load(self.get_filename(), sr = rate,  offset = start_edit, duration = duration)
        data_noise, sr_noise = librosa.load(self.get_filename(), sr = rate, offset = end, duration = duration)
        librosa.output.write_wav(self.get_output_file(), data, sr)
        librosa.output.write_wav(self.get_output_file_noise(), data_noise, sr_noise)
        return 0

    def read_text_descriptor_file(self, text_file):
            """
            Reads log files related to the wave file
            Inputs:
            text_file
            
            Outputs:
            data
            """
            
            data = pd.read_csv(text_file, delim_whitespace = True)
            data = data.dropna(axis = 1, how = 'any')
            
            #data = data.drop(labels=["start time", "end time"], axis = 0, inplace = True)
            
        
            
            return data

    def audio_creation(self, outputfilename, df3):
        """
        Reads in the start and end time of each snap shot gets the audio snippet and saves the output to a new audio file.

        """
        output_file_list = []
        output_file_list_noise = []
        for start, end in zip(df3["start time"], df3["end time"]):
        #start_vals.append(start)
            
            
            dirname_strip = outputfilename.strip('.wav')
            
            output_file_list.append(dirname_strip)
            output_file_list.append("_")
            output_file_list.append(str(start))
            output_file_list.append('.wav')
            output_file = ''.join(output_file_list)
            self.set_output_file(output_file)
            output_file_list = []
            
            output_file_list.append(dirname_strip)
            output_file_list.append("_")
            output_file_list.append(str(end))
            output_file_list.append('_noise.wav')
            output_file_noise = ''.join(output_file_list)
            self.set_output_file_noise(output_file_noise)
            output_file_list = []        



            self.get_and_save_audio(start, end)
        return 0


    def audio_stripper(self):
        count = []
        location = filedialog.askdirectory()
        exclude = set(['modified', 'Berardius_Baja-Annotated'])
        #allows exclusion of bad folders and modified files
        for root, dirs, files in os.walk(location, topdown=True):
            [dirs.remove(d) for d in list(dirs) if d in exclude]                    
            for file in files:
                #print(os.path.join(root, file))
                if file.endswith(".wav"):
                    filename = os.path.join(root, file)
                    self.set_filename(filename)
                    outputfolder = root + '/modified'
                    if not os.path.exists(outputfolder):
                        os.makedirs(outputfolder)

                    outputfilename = os.path.join(outputfolder, file)
                    directory = os.path.dirname(self.get_filename())
  
                         
                    dirname_remove = filename.split('/')[-1]
                    print("root = ", root)
                    name = file.split('.') #filename without extension
                    filepath = directory  

                    #Variable initialisation
                    a = []
                    b = []
                    c = []
                    header = ["start time", "end time"]

                    df3 = pd.DataFrame()
                    df = pd.DataFrame()

                    #pattern to match wave file against        
                    pattern = name[0] + '*.log'
                    
                    #need to remove folder location
                    print(filepath)
                    for f in os.listdir(filepath):   
                        a.append(fnmatch.fnmatch(f, pattern)) 
                        print(a, "a rolling")      
                        b.append(f)
                    #merges match and filename to one df
                    print(a, "a")
                    df = pd.DataFrame({'match': a,
                                        'filename': b})  

                    print(df)
                    #finds all files in df that pattern match
                    vals = df.loc[(df["match"]==True)]
                    if vals.empty == False:
                                      
                        #print(vals)
                        #reads text in items and appends it to dataframe.
                                              
                        for items in vals["filename"]:
                            text_file = filepath + '/' + items
                            c.append(self.read_text_descriptor_file(text_file))
                            print(c)
                        df3 = df3.append(c)   
                        df3.columns = header
                                #match_df = match_text(directory, name2)
                        self.audio_creation(outputfilename, df3)
                    else:
                        break









