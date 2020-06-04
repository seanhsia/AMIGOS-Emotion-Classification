# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 23:40:59 2020

@author: USER
"""

import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pickle

import Models as m
from sklearn.model_selection import train_test_split

def LoadTrainTestData():
    with open('./tmp/data_list_preprocess_test.pkl', 'rb') as file:
           data_array= np.array(pickle.load(file))
    #groundtruth=(pd.read_csv('./groundtruth_dropnan.csv').loc[:, ['Valence', 'Arousal']].to_numpy()-1)/8
    groundtruth_index=pd.read_csv('./groundtruth_dropnan_EEG.csv').loc[:, ['Valence_Index', 'Arousal_Index']].to_numpy()
    
    #labels=np.concatenate((groundtruth, groundtruth_index), axis=1)
    labels=groundtruth_index
    X_train, X_test, y_train, y_test=train_test_split(data_array, labels, test_size=0.2, random_state=3)
    return X_train, X_test, y_train, y_test

def TrainAutoEncoder(train_loader, test_loader, model, epochs, device, optimizer='Adam', lr=1e-4):
    print("=======================START AUTOENCODER TRAINING=======================")
    #print(model)
    
    
    train_loss=list()
    test_loss=list()
    
    model=model.to(device)
    loss_fct=nn.BCELoss()
    optimizer=torch.optim.Adam(model.parameters(), lr)
    
    for epoch in range(epochs):
        model.train()
        total_loss=0
        for step, (batch_x, batch_y) in enumerate(train_loader):
            X_hat=model(batch_x)
            loss = loss_fct(X_hat, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
        train_loss.append(total_loss/len(train_loader.dataset))
        test_loss.append(TestAutoEncoder(test_loader, model, loss_fct).item())
        
        if epoch != 0 and min(train_loss) == (total_loss/len(train_loader.dataset)):
            torch.save(model.state_dict(), './modelweight/AutoEncoder_lowestloss_ica_energy')
                
        if (epoch+1) % 10 == 0:
            print("Epoch: ", epoch+1, "| Loss: ", train_loss[-1])
            print("Test Loss: ", test_loss[-1])

        if (epoch+1) % 100 ==0:
            torch.save(model.state_dict(), './modelweight/AutoEncoder_'+str(epoch+1)+"_ica_energy")
    return train_loss, test_loss
    
def TestAutoEncoder(test_loader, model, loss_fct):
    model.eval()
    total_loss=0
    for x_batch, y_batch in test_loader:
        with torch.no_grad():
            X_hat = model(x_batch)
            total_loss += loss_fct(X_hat , y_batch) * x_batch.size(0)
    return total_loss/len(test_loader.dataset)


def TrainClassifier(train_loader, test_loader, model, epochs, device, optimizer='Adam', lr=1e-2):
    print("=======================START Classifier TRAINING=======================")
    #print(model)
    
    train_loss=list()
    test_loss=list()
    train_acc=list()
    test_acc=list()
    valence_acc=list()
    arousal_acc=list()
    

    
    model=model.to(device)
    loss_fct=nn.BCELoss()
    optimizer=torch.optim.Adam(model.parameters(), lr, weight_decay=1e-5)
    
    for epoch in range(epochs):
        model.train()
        total_loss=0
        train_correct=0
        for step, (batch_x, batch_y) in enumerate(train_loader):
            y_pred=model(batch_x)
            loss = loss_fct(y_pred, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            
            pred = (y_pred.data > 0.5).float()
            train_correct += pred.eq(batch_y.data).cpu().sum(1).eq(2).sum().item()
            
            loss.backward()
                   
            optimizer.step()
            optimizer.zero_grad()
            
        train_loss.append(total_loss/len(train_loader.dataset))
        train_acc.append(train_correct/len(train_loader.dataset))
        
        testloss, testacc, valenceacc, arousalacc=TestClassifier(model, test_loader, loss_fct)
        valence_acc.append(valenceacc)
        arousal_acc.append(arousalacc)
        
        
        test_loss.append(testloss.item())
        test_acc.append(testacc)
        if epoch != 0 and max(test_acc) == (testacc):
            torch.save(model.state_dict(), './modelweight/Classifier_best_ica_energy')    
        if (epoch+1) % 10 == 0:
            print("Epoch: ", epoch+1, "| Loss: ", train_loss[-1], "| Accuracy: ", train_acc[-1])
            print("Test Loss: ", test_loss[-1], "| Accuracy: ", test_acc[-1])

    return train_loss, test_loss, train_acc, test_acc, valence_acc, arousal_acc

def TestClassifier(model, test_loader, loss_fct):
    model.eval()
    total_loss=0
    test_correct=0
    v_correct=0
    a_correct=0
    for batch_x, batch_y in test_loader:
        with torch.no_grad():
            y_pred = model(batch_x)
            total_loss += loss_fct(y_pred, batch_y) * batch_x.size(0)
            pred = (y_pred.data > 0.5).float()
            test_correct += pred.eq(batch_y.data).cpu().sum(1).eq(2).sum().item()
            v_correct += pred.eq(batch_y.data).cpu().sum(0)[0].item()
            a_correct += pred.eq(batch_y.data).cpu().sum(0)[1].item()
    #print("Valence: ", v_correct/len(test_loader.dataset), "Arousal: ", a_correct/len(test_loader.dataset))
    return total_loss/len(test_loader.dataset), test_correct/len(test_loader.dataset), v_correct/len(test_loader.dataset), a_correct/len(test_loader.dataset)
    


if __name__=='__main__':
    #hyperparameters
    epochs=5000
    device=torch.device('cuda:0')
    batch_size=16
    autoencoder_weight_path='./modelweight/AutoEncoder_lowestloss_ica_energy'
    clf_weight_path='./modelweight/Classifier_500BCE'
    train_autoencoder=False
    train_classifier=True
    
    #Load data
    X_train, X_test, y_train,y_test = LoadTrainTestData()
    
    #To torch tensor
    X_train = torch.as_tensor(X_train, dtype=torch.float, device=device)
    y_train = torch.as_tensor(y_train, dtype=torch.float, device=device)
    X_test = torch.as_tensor(X_test, dtype=torch.float, device=device)
    y_test = torch.as_tensor(y_test, dtype=torch.float, device=device)
    
    #devided into batch_size
    if train_autoencoder:
        train_loader = DataLoader(TensorDataset(X_train, X_train), 
                              batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, X_test), 
                             batch_size=batch_size, shuffle=True)        
        autoencoder=m.AutoEncoderEEGDWT2D()
        train_loss, test_loss=TrainAutoEncoder(train_loader, test_loader, autoencoder, epochs, device)
        
    elif train_classifier:
        train_loader = DataLoader(TensorDataset(X_train, y_train), 
                              batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test), 
                             batch_size=batch_size, shuffle=True)
    
        autoencoder=m.AutoEncoderEEGDWT2D()
        if autoencoder_weight_path != "":
            autoencoder.load_state_dict(torch.load(autoencoder_weight_path))
    
        clf=m.ClassifierEEGDWT2D(autoencoder.encoder)
        train_loss, test_loss, train_acc, test_acc, valence_acc, arousal_acc=TrainClassifier(
                train_loader, test_loader, clf, epochs, device)
    