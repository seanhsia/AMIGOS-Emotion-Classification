# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 18:21:31 2020

@author: USER
"""

#load data
import pickle

#math tools
import numpy as np

#Normalize
from sklearn.preprocessing import MinMaxScaler

#computing entropy
from scipy.stats import entropy

#compute discrete wavelet transform
import pywt




def LoadProcessedData():
    with open('./tmp/data_list_ica.pkl', 'rb') as file:
        data_list=pickle.load(file)
    return data_list

def Energy(signal):
    return np.diag(np.dot(signal, signal.T))


def DiscreteWaveletTransform(signal, feature, windowsize=4, sampling_rate=128):
    samples=int(sampling_rate * windowsize / 2)
    n=signal.shape[1]
    
    scaler=MinMaxScaler()
    
    if feature == "entropy":
        for i in range((samples), (n-samples), samples):
            lowpass_signal, noise=pywt.dwt(signal[:,i-samples:i+samples], 'db4')
            for j in range(4):
                lowpass_signal, highpass_signal=pywt.dwt(lowpass_signal, 'db4')
                highpass_signal=scaler.fit_transform(highpass_signal.T).T
                if j==0:
                    if i == samples:
                        gamma=entropy(highpass_signal.T).reshape(1,-1).T
                    else:
                        gamma=np.concatenate((gamma, 
                                    entropy(highpass_signal.T).reshape(1,-1).T), axis=1)
    
                if j==1:
                    if i == samples:
                        beta=entropy(highpass_signal.T).reshape(1,-1).T
                    else:
                        beta=np.concatenate((beta, 
                                    entropy(highpass_signal.T).reshape(1,-1).T), axis=1)                
                if j==2:
                    if i == samples:
                        alpha=entropy(highpass_signal.T).reshape(1,-1).T
                    else:
                        alpha=np.concatenate((alpha, 
                                    entropy(highpass_signal.T).reshape(1,-1).T), axis=1)
    
                if j==3:
                    if i == samples:
                        theta=entropy(highpass_signal.T).reshape(1,-1).T
                    else:
                        theta=np.concatenate((theta, 
                                    entropy(highpass_signal.T).reshape(1,-1).T), axis=1)
    
    elif feature == "energy":
        for i in range((samples), (n-samples), samples):
            lowpass_signal, noise=pywt.dwt(signal[:,i-samples:i+samples], 'db4')
            for j in range(4):
                lowpass_signal, highpass_signal=pywt.dwt(lowpass_signal, 'db4')
                #highpass_signal=scaler.fit_transform(highpass_signal.T).T
                if j==0:
                    if i == samples:
                        gamma=Energy(highpass_signal).reshape(1,-1).T
                    else:
                        gamma=np.concatenate((gamma, 
                                    Energy(highpass_signal).reshape(1,-1).T), axis=1)
    
                if j==1:
                    if i == samples:
                        beta=Energy(highpass_signal).reshape(1,-1).T
                    else:
                        beta=np.concatenate((beta, 
                                    Energy(highpass_signal).reshape(1,-1).T), axis=1)                
                if j==2:
                    if i == samples:
                        alpha=Energy(highpass_signal).reshape(1,-1).T
                    else:
                        alpha=np.concatenate((alpha, 
                                    Energy(highpass_signal).reshape(1,-1).T), axis=1)
    
                if j==3:
                    if i == samples:
                        theta=Energy(highpass_signal).reshape(1,-1).T
                    else:
                        theta=np.concatenate((theta, 
                                    Energy(highpass_signal).reshape(1,-1).T), axis=1)
            
    gamma=scaler.fit_transform(gamma.T).T
    beta=scaler.fit_transform(beta.T).T
    alpha=scaler.fit_transform(alpha.T).T
    theta=scaler.fit_transform(theta.T).T
    
    return np.array([gamma, beta, alpha, theta])


        
               
        
def ZeroPadding(EEG_list, total_length):
#Padding with value 0
    data_list=list()
    for i in range(len(EEG_list)):
        EEG_pad = np.pad(EEG_list[i], ((0,0),(0,0), (0, total_length-EEG_list[i].shape[2])), 'constant')
        data_list.append(EEG_pad)
    return data_list



if __name__ == '__main__':
    feature="entropy"
    
    data_list=LoadProcessedData()
    data_list=[DiscreteWaveletTransform(eeg, feature) for eeg in data_list]
    data_list=ZeroPadding(data_list, 80)
    
   
    with open("./tmp/data_list_preprocess_test.pkl", "wb") as file:
         pickle.dump(data_list, file)

        