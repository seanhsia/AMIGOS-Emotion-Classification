# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:19:37 2020

@author: USER
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, dropout=0.5):
        super(Encoder, self).__init__()
        '''
            EEG CNN encoder
        '''
        self.EEG_conv1=nn.Sequential(
                nn.Conv1d(
                        in_channels=14,
                        out_channels=32,
                        kernel_size=15,
                        stride=1,
                        padding=7
                    ),
               nn.BatchNorm1d(32), 
               nn.ReLU(True)
            )
        #32x20000
        
        self.EEG_pool1=nn.MaxPool1d(kernel_size=5, return_indices=True)#32x4000
        
        self.EEG_dropout1=nn.Dropout(p=dropout)
        
        self.EEG_conv2=nn.Sequential(
               nn.Conv1d(
                       in_channels=32,
                       out_channels=64,
                       kernel_size=5,
                       stride=1,
                       padding=2
                    ),
               nn.BatchNorm1d(64), 
               nn.ReLU(True)
            )
        #64x4000
        
        self.EEG_pool2=nn.MaxPool1d(kernel_size=2, return_indices=True)#64x2000
        
        self.EEG_dropout2=nn.Dropout(p=dropout)
        
        self.EEG_conv3=nn.Sequential(
               nn.Conv1d(
                       in_channels=64,
                       out_channels=64,
                       kernel_size=3,
                       stride=1,
                       padding=1
                    ),#64*2000
               nn.BatchNorm1d(64), 
               nn.ReLU(True)
            )
        #64x2000
        
        self.EEG_pool3=nn.MaxPool1d(kernel_size=2, return_indices=True)#64x1000
        self.EEG_dropout3=nn.Dropout(p=dropout)
        
        '''
            EEG CNN encoder
        '''        
        self.ECG_conv1=nn.Sequential(
                nn.Conv1d(
                        in_channels=2,
                        out_channels=32,
                        kernel_size=15,
                        stride=1,
                        padding=7
                    ),
               nn.BatchNorm1d(32), 
               nn.ReLU(True)
            )
        #32x20000
        
        self.ECG_pool1=nn.MaxPool1d(kernel_size=5, return_indices=True)#32x4000
        
        self.ECG_dropout1=nn.Dropout(p=dropout)
        
        self.ECG_conv2=nn.Sequential(
               nn.Conv1d(
                       in_channels=32,
                       out_channels=64,
                       kernel_size=5,
                       stride=1,
                       padding=2
                    ),
               nn.BatchNorm1d(64), 
               nn.ReLU(True)
            )
        #64x4000
        
        self.ECG_pool2=nn.MaxPool1d(kernel_size=2, return_indices=True)#64x2000
        
        self.ECG_dropout2=nn.Dropout(p=dropout)
        
        self.ECG_conv3=nn.Sequential(
               nn.Conv1d(
                       in_channels=64,
                       out_channels=64,
                       kernel_size=3,
                       stride=1,
                       padding=1
                    ),#64*2000
               nn.BatchNorm1d(64), 
               nn.ReLU(True)
            )
        #64x2000
        
        self.ECG_pool3=nn.MaxPool1d(kernel_size=2, return_indices=True)#64x1000
        self.ECG_dropout3=nn.Dropout(p=dropout)
        
            
        '''
             GSR CNN encoder
        '''            
        self.GSR_conv1=nn.Sequential(
                nn.Conv1d(
                        in_channels=1,
                        out_channels=32,
                        kernel_size=9,
                        stride=1,
                        padding=4
                    ),#32x20000
               nn.BatchNorm1d(32), 
               nn.ReLU(True)
            )
        self.GSR_pool1=nn.MaxPool1d(kernel_size=2, return_indices=True)
        self.GSR_dropout1=nn.Dropout(p=dropout)
        
        self.GSR_conv2=nn.Sequential(
               nn.Conv1d(
                       in_channels=32,
                       out_channels=64,
                       kernel_size=3,
                       stride=1,
                       padding=1
                    ),#64x10000
               nn.BatchNorm1d(64), 
               nn.ReLU(True)
            )
        self.GSR_pool2=nn.MaxPool1d(kernel_size=1, return_indices=True)
        self.GSR_dropout2=nn.Dropout(p=dropout)
               
        self.GSR_conv3=nn.Sequential(
                nn.Conv1d(
                       in_channels=64,
                       out_channels=64,
                       kernel_size=1,
                       stride=1,
                       padding=0
                    ),#64x10000
               nn.BatchNorm1d(64), 
               nn.ReLU(True)
            )
        self.GSR_pool3=nn.MaxPool1d(kernel_size=1, return_indices=True)
        self.GSR_dropout3=nn.Dropout(p=dropout)
            
               
    def forward(self, X):
        EEG=X[:,0:14,:]
        ECG=X[:,14:16,:]
        GSR=X[:,16,:].view(X.size(0),1,-1)
        '''
            EEG forward pass
        '''
        X=self.EEG_conv1(EEG)
        X, EEG_indicies1=self.EEG_pool1(X)
        X=self.EEG_dropout1(X)
        
        X=self.EEG_conv2(X)
        X, EEG_indicies2=self.EEG_pool2(X)
        X=self.EEG_dropout2(X)
        
        X=self.EEG_conv3(X)
        X, EEG_indicies3=self.EEG_pool3(X)
        EEG_code=self.EEG_dropout3(X)
        
        EEG_indicies_tuple=(EEG_indicies3, EEG_indicies2, EEG_indicies1)
        
        '''
            ECG forward pass
        '''        
        X=self.ECG_conv1(ECG)
        X, ECG_indicies1=self.ECG_pool1(X)
        X=self.ECG_dropout1(X)
        
        X=self.ECG_conv2(X)
        X, ECG_indicies2=self.ECG_pool2(X)
        X=self.ECG_dropout2(X)
        
        X=self.ECG_conv3(X)
        X, ECG_indicies3=self.ECG_pool3(X)
        ECG_code=self.ECG_dropout3(X)
        
        ECG_indicies_tuple=(ECG_indicies3, ECG_indicies2, ECG_indicies1)
        
        '''
            GSR forward pass
        '''
        X=self.GSR_conv1(GSR)
        X, GSR_indicies1=self.GSR_pool1(X)
        X=self.GSR_dropout1(X)
        X=self.GSR_conv2(X)
        X, GSR_indicies2=self.GSR_pool2(X)
        X=self.GSR_dropout2(X)
        
        X=self.GSR_conv3(X)
        X, GSR_indicies3=self.GSR_pool3(X)
        GSR_code=self.GSR_dropout3(X)


        GSR_indicies_tuple=(GSR_indicies3, GSR_indicies2, GSR_indicies1)        
        return EEG_code, ECG_code, GSR_code, EEG_indicies_tuple, ECG_indicies_tuple,GSR_indicies_tuple
    
class Decoder(nn.Module):
    def __init__(self, dropout=0.5):
        super(Decoder,self).__init__()
        self.EEG_deconv1=nn.Sequential(
                nn.ConvTranspose1d(
                        in_channels=64,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=1
                    ),
                nn.BatchNorm1d(64),
                nn.ReLU(True)
            )
        self.EEG_unpool1=nn.MaxUnpool1d(kernel_size=2)
        self.EEG_dropout1=nn.Dropout(p=dropout)
        
        self.EEG_deconv2=nn.Sequential(        
                nn.ConvTranspose1d(
                        in_channels=64,
                        out_channels=32,
                        kernel_size=5,
                        stride=1,
                        padding=2
                    ),
                nn.BatchNorm1d(32),
                nn.ReLU(True)
            )
        self.EEG_unpool2=nn.MaxUnpool1d(kernel_size=2)
        self.EEG_dropout2=nn.Dropout(p=dropout)
                
        self.EEG_deconv3=nn.Sequential(
                nn.ConvTranspose1d(
                        in_channels=32,
                        out_channels=14,
                        kernel_size=15,
                        stride=1,
                        padding=7
                    ),
                nn.BatchNorm1d(14),
                nn.Sigmoid()
            )
        self.EEG_unpool3=nn.MaxUnpool1d(kernel_size=5)
        self.EEG_dropout3=nn.Dropout(p=dropout)
        
        
        self.ECG_deconv1=nn.Sequential(
                nn.ConvTranspose1d(
                        in_channels=64,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=1
                    ),
                nn.BatchNorm1d(64),
                nn.ReLU(True)
            )
        self.ECG_unpool1=nn.MaxUnpool1d(kernel_size=2)
        self.ECG_dropout1=nn.Dropout(p=dropout)
        
        self.ECG_deconv2=nn.Sequential(        
                nn.ConvTranspose1d(
                        in_channels=64,
                        out_channels=32,
                        kernel_size=5,
                        stride=1,
                        padding=2
                    ),
                nn.BatchNorm1d(32),
                nn.ReLU(True)
            )
        self.ECG_unpool2=nn.MaxUnpool1d(kernel_size=2)
        self.ECG_dropout2=nn.Dropout(p=dropout)
                
        self.ECG_deconv3=nn.Sequential(
                nn.ConvTranspose1d(
                        in_channels=32,
                        out_channels=2,
                        kernel_size=15,
                        stride=1,
                        padding=7
                    ),
                nn.BatchNorm1d(2),
                nn.Sigmoid()
            )
        self.ECG_unpool3=nn.MaxUnpool1d(kernel_size=5)
        self.ECG_dropout3=nn.Dropout(p=dropout)
            
               
        self.GSR_deconv1=nn.Sequential(
                nn.ConvTranspose1d(
                        in_channels=64,
                        out_channels=64,
                        kernel_size=1,
                        stride=1,
                        padding=0
                    ),
                nn.BatchNorm1d(64),
                nn.ReLU(True)
            )
        self.GSR_unpool1=nn.MaxUnpool1d(kernel_size=1)
        self.GSR_dropout1=nn.Dropout(p=dropout)
        
        self.GSR_deconv2=nn.Sequential(        
                nn.ConvTranspose1d(
                        in_channels=64,
                        out_channels=32,
                        kernel_size=3,
                        stride=1,
                        padding=1
                    ),
                nn.BatchNorm1d(32),
                nn.ReLU(True)
            )
        self.GSR_unpool2=nn.MaxUnpool1d(kernel_size=1)
        self.GSR_dropout2=nn.Dropout(p=dropout)
        
        self.GSR_deconv3=nn.Sequential(        
                nn.ConvTranspose1d(
                        in_channels=32,
                        out_channels=1,
                        kernel_size=9,
                        stride=1,
                        padding=4
                    ),
                nn.BatchNorm1d(1),
                nn.Sigmoid()
            )
        self.GSR_unpool3=nn.MaxUnpool1d(kernel_size=2)
        self.GSR_dropout3=nn.Dropout(p=dropout)
            
                
    def forward(self, EEG_code, ECG_code, GSR_code, EEG_indicies_tuple, ECG_indicies_tuple,GSR_indicies_tuple):
        EEG_code=self.EEG_unpool1(EEG_code, EEG_indicies_tuple[0])
        EEG_code=self.EEG_deconv1(EEG_code)
        EEG_code=self.EEG_dropout1(EEG_code)
        
        EEG_code=self.EEG_unpool2(EEG_code, EEG_indicies_tuple[1])        
        EEG_code=self.EEG_deconv2(EEG_code)
        EEG_code=self.EEG_dropout2(EEG_code)
        
        EEG_code=self.EEG_unpool3(EEG_code, EEG_indicies_tuple[2])
        EEG_decode=self.EEG_deconv3(EEG_code)
        #EEG_decode=self.EEG_dropout3(EEG_code)
        
        ECG_code=self.ECG_unpool1(ECG_code, ECG_indicies_tuple[0])
        ECG_code=self.ECG_deconv1(ECG_code)
        ECG_code=self.ECG_dropout1(ECG_code)
        
        ECG_code=self.ECG_unpool2(ECG_code, ECG_indicies_tuple[1])        
        ECG_code=self.ECG_deconv2(ECG_code)
        ECG_code=self.ECG_dropout2(ECG_code)
        
        ECG_code=self.ECG_unpool3(ECG_code, ECG_indicies_tuple[2])
        ECG_decode=self.ECG_deconv3(ECG_code)
        #ECG_decode=self.ECG_dropout3(ECG_code)

        GSR_code=self.GSR_unpool1(GSR_code, GSR_indicies_tuple[0])        
        GSR_code=self.GSR_deconv1(GSR_code)
        GSR_code=self.GSR_dropout1(GSR_code)

        GSR_code=self.GSR_unpool2(GSR_code, GSR_indicies_tuple[1])
        GSR_code=self.GSR_deconv2(GSR_code)
        GSR_code=self.GSR_dropout2(GSR_code)


        GSR_code=self.GSR_unpool3(GSR_code, GSR_indicies_tuple[2])
        GSR_decode=self.GSR_deconv3(GSR_code)
        #GSR_decode=self.GSR_dropout3(GSR_code) 
        
        X_decode=torch.cat((EEG_decode, ECG_decode, GSR_decode), dim=1)
        return X_decode
    
class Classifier(nn.Module):
    def __init__(self, encoder, dropout=0.5):
        super(Classifier, self).__init__()
        self.encoder=encoder
        '''
        self.conv_EEG=nn.Conv1d(
                    in_channels=64,
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0)
        self.conv_ECG=nn.Conv1d(
                    in_channels=64,
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0)
        self.conv_GSR=nn.Conv1d(
                    in_channels=64,
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0)
        '''
        self.GAP_EEG=nn.AdaptiveAvgPool1d(32)
        self.GAP_ECG=nn.AdaptiveAvgPool1d(32)
        self.GAP_GSR=nn.AdaptiveAvgPool1d(64)
        self.classifier=nn.Sequential(
                nn.Linear(8192,1024),
                nn.ReLU(True),
                nn.Linear(1024,256),
                nn.ReLU(True),
                nn.Linear(256,2),
                nn.Sigmoid()
            )
        
    def forward(self, X):
        EEG_code, ECG_code, GSR_code,EEG_indicies_tuple, ECG_indicies_tuple, GSR_indicies_tuple=self.encoder(X)
        #EEG_feat=self.conv_EEG(EEG_code)
        #ECG_feat=self.conv_ECG(ECG_code)
        #GSR_feat=self.conv_GSR(GSR_code)
        
        #Global Average Pooling and Flatten
        EEG_feat=self.GAP_EEG(EEG_code).view(EEG_code.size(0),-1)
        ECG_feat=self.GAP_ECG(ECG_code).view(ECG_code.size(0), -1)
        GSR_feat=self.GAP_GSR(GSR_code).view(GSR_code.size(0), -1)
        Bio_feat=torch.cat((EEG_feat, ECG_feat, GSR_feat), dim=1)
        
        #Pass FCN
        pred=self.classifier(Bio_feat)
        
        return pred
        
    
class CNNAutoEncoder(nn.Module):
    def __init__(self, dropout=0.5):
        super(CNNAutoEncoder, self).__init__()
        self.encoder=Encoder(dropout=dropout)
        self.decoder=Decoder(dropout=dropout)
        
    def forward(self, X):
        EEG_code, ECG_code, GSR_code, EEG_indicies_tuple, ECG_indicies_tuple, GSR_indicies_tuple=self.encoder(X)
        X_hat=self.decoder(EEG_code, ECG_code, GSR_code, EEG_indicies_tuple, ECG_indicies_tuple, GSR_indicies_tuple)
        return X_hat
    
    
    
    
    
class EncoderEEGDWT2D(nn.Module):
    def __init__(self, dropout=0.5):
        super(EncoderEEGDWT2D, self).__init__()
        self.conv1=nn.Sequential(
                    nn.Conv2d(in_channels=4,
                              out_channels=16,
                              kernel_size=(1,5),
                              stride=(1,1),
                              padding=(0,2),
                              bias=False),
                    nn.BatchNorm2d(16)
                    )
        self.conv2=nn.Sequential(
                    nn.Conv2d(in_channels=16,
                              out_channels=32,
                              kernel_size=(2,1),
                              stride=(1,1),
                              groups=16,
                              bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(True))
        self.pool1=nn.MaxPool2d(kernel_size=(1,4),
                                stride=(1,4),
                                return_indices=True)
        self.dropout1=nn.Dropout2d(p=dropout)
        #Separable Convolution Layer
        self.sep_conv = nn.Sequential(
                    #Convolution
                    nn.Conv2d(
                        in_channels = 32,
                        out_channels = 32,
                        kernel_size = (1, 15),
                        stride = (1, 1),
                        padding = (0, 7),
                        bias = False,
                    ),
        
                    #Batch Normalization
                    nn.BatchNorm2d(32),
                    #Activation Function
                    nn.ReLU(True))
                    
        #Pooling Layer
        self.pool2=nn.MaxPool2d(
                    kernel_size = (1, 4),
                    stride = (1, 4),
                    return_indices=True)
        
                    #Dropout
        self.dropout2=nn.Dropout(p = dropout)
        
    def forward(self, X):
        feat=self.conv1(X)
        
        feat=self.conv2(feat)
        feat, indicies1=self.pool1(feat)
        feat=self.dropout1(feat)
        
        feat=self.sep_conv(feat)
        feat, indicies2=self.pool2(feat)
        feat=self.dropout2(feat)
        
        return feat, [indicies1, indicies2]
    
    
    
class DecoderEEGDWT2D(nn.Module):
    def __init__(self, dropout=0.5):
        super(DecoderEEGDWT2D, self).__init__()
        
        self.deconv1=nn.Sequential(
                        nn.ConvTranspose2d(in_channels=32,
                                           out_channels=32,
                                           kernel_size = (1, 15),
                                           stride = (1, 1),
                                           padding = (0, 7),
                                           bias = False),
                        nn.BatchNorm2d(32),
                        nn.ReLU(True)
                    )
        self.unpool1=nn.MaxUnpool2d(kernel_size=(1,4),
                                    stride=(1,4))
        self.dropout1=nn.Dropout(p=dropout)
        
        self.deconv2=nn.Sequential(
                        nn.ConvTranspose2d(in_channels=32,
                                           out_channels=16,
                                           kernel_size=(2,1),
                                           stride=(1,1),
                                           groups=16,
                                           bias=False),
                        nn.BatchNorm2d(16),
                        nn.ReLU(True)
                    )
        self.unpool2=nn.MaxUnpool2d(kernel_size=(1,4),
                                    stride=(1,4))
        self.dropout2=nn.Dropout(p=dropout)
        
        self.deconv3=nn.Sequential(
                        nn.ConvTranspose2d(in_channels=16,
                                           out_channels=4,
                                           kernel_size=(1,5),
                                           stride=(1,1),
                                           padding=(0,2),
                                           bias=False),
                        nn.BatchNorm2d(4),
                        nn.Sigmoid()
                    )
                        
                        
    def forward(self, feat, indicies_list):     
        feat=self.unpool1(feat, indicies_list[1])
        feat=self.deconv1(feat)
        feat=self.dropout1(feat)
        feat=self.unpool2(feat, indicies_list[0])
        feat=self.deconv2(feat)
        feat=self.dropout2(feat)
        X_hat=self.deconv3(feat)
        return X_hat
    
    
class AutoEncoderEEGDWT2D(nn.Module):
    def __init__(self, dropout=0.5):
        super(AutoEncoderEEGDWT2D, self).__init__()
        self.encoder=EncoderEEGDWT2D(dropout)
        self.decoder=DecoderEEGDWT2D(dropout)
    
    def forward(self, X):
        code, indicies=self.encoder(X)
        X_hat=self.decoder(code, indicies)
        return X_hat
    
class ClassifierEEGDWT2D(nn.Module):
    def __init__(self, encoder):
        super(ClassifierEEGDWT2D, self).__init__()
        self.encoder=encoder
        self.clf=nn.Sequential(
                    nn.Linear(2080,512),
                    nn.ReLU(True),
                    nn.Dropout(0.5),
                    nn.Linear(512, 64),
                    nn.ReLU(True),
                    nn.Dropout(0.5),
                    nn.Linear(64,2),
                    nn.Sigmoid())
        
    def forward(self, X):
        feat, indicies=self.encoder(X)
        pred=self.clf(feat.reshape(feat.shape[0],-1))
        return pred
    
    
