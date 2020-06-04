# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:39:57 2020

@author: USER
"""
import pickle
import numpy as np
import pandas as pd

import mne

from sklearn.preprocessing import MinMaxScaler

def PreprocessData():
    with open("./tmp/data_list.pkl", "rb") as file:
        data_list=pickle.load(file)
    
    EEG_list=list()

        
    #removing first  3secs
    for tester in data_list:
        for instance in tester:
            EEG_list.append(instance[0:14][:,384:])
            
    #record the indecies of nan array
    nan_set=set() 
    
    #Preprocess EEG 
    for i in range(len(EEG_list)):
        if(np.all(np.isnan(EEG_list[i]))):
            nan_set.add(i)
            EEG_list[i]=np.nan
    

    
    #Remove the data with Nan values
    EEG_list=np.delete(EEG_list,list(nan_set)).tolist()
    pd.read_csv("./groundtruth.csv").drop(list(nan_set)).to_csv("./groundtruth_dropnan_EEG.csv", index=False)
    return EEG_list

def ICA(EEG_list, index):
    i=index
    EEG_list=EEG_list[index:]
    scaler=MinMaxScaler()
    
    for eeg in EEG_list:
        info = mne.create_info(14, 128, ch_types=['eeg']*14)
        eeg = scaler.fit_transform(eeg.T).T * 10e-5
        
        raw = mne.io.RawArray(eeg, info)
        raw_tmp = raw.copy()
        raw_tmp.filter(1, None)
        
        ica=mne.preprocessing.ICA(method="infomax",
                                  fit_params={"extended": True},
                                  random_state=1)
        
        ica.fit(raw_tmp)
        ica.plot_sources(inst=raw_tmp,start=0, stop=40)
        
        raw_corrected=raw.copy()
        channel_exclude=list(map(int,
                                 input("Input the channel index you want to exclude: ").split()))

        ica.exclude=channel_exclude
        ica.apply(raw_corrected)        
        raw_corrected.plot(n_channels=14, start=0, duration=40)
        
        eeg_corrected=scaler.fit_transform(raw_corrected[:][0].T).T
        
        
        with open("./eeg_ica/eeg_ica_"+str(i), "wb") as file:
            pickle.dump(eeg_corrected, file)
        
        i+=1

def LoadICAData():
    load_path="./eeg_ica/eeg_ica_"
    save_path="./tmp/data_list_ica.pkl"
    data_list=list()
    for i in range(614):
        with open(load_path+str(i), 'rb') as file:
            data_list.append(pickle.load(file))
            
    with open(save_path, 'wb') as file:
        pickle.dump(data_list,file)
            

if __name__=='__main__':
    EEG_list=PreprocessData()
    index=int(input("Input the starting index: "))
    ICA(EEG_list, index)
    LoadICAData()