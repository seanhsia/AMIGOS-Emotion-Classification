# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 00:32:08 2020

@author: USER
"""

# drawing packages
import matplotlib.pyplot as plt
import seaborn as sns

# math packages
import numpy as np
import pandas as pd

#read .mat file
import scipy.io

#save data as pickle
import pickle

#k-means
from sklearn.cluster import KMeans



def GetGroundtruth_Df(selfassessment_list, pseudo_label, pseudo_index, plot_ground_video=False, plot_individual=False):
    valence=list()
    arousal=list()
    videos=list()
    quadrants=list()
    valence_index=list()
    arousal_index=list()
    video=["August Rush", "Love Actually", "The Thin Red Line", "House of Flying Daggers",
            "Exocist", "My girl", "My Bodyguard", "Silent Hill", "Prestige", "Pink Flammingos",
            "Black Swan", "Airplane", "When Harry Met Sally", "Mr Beans Holiday",
            "Love Actually", "Hot shots"]
    quadrant=["LAHV", "LAHV", "LALV", "LAHV", "LALV", "LALV", "LALV", "HALV", "HALV", 
              "HALV", "HALV", "HAHV", "HAHV", "LAHV", "HAHV", "HAHV"]

    for selfassessment in selfassessment_list:
        videos.extend(video)
        quadrants.extend(quadrant)
        for instance in selfassessment:
            arousal.append(instance[0])
            valence.append(instance[1])
    
    for index in pseudo_index:
        if index == 0:
            valence_index.append(0)
            arousal_index.append(0)
        elif index == 1:
            valence_index.append(1)
            arousal_index.append(0)
        elif index == 2:
            valence_index.append(0)
            arousal_index.append(1)
        elif index == 3:
            valence_index.append(1)
            arousal_index.append(1)
    
    groundtruth_df = pd.DataFrame(
                {
                    "Valence": valence,
                    "Arousal": arousal,
                    "Videos": videos,
                    "Quadrants": quadrants,
                    "Pseudo_Label": pseudo_label,
                    "Pseudo_Index": pseudo_index,
                    "Valence_Index": valence_index,
                    "Arousal_Index": arousal_index
                }
            )
    
    #plot ground_label_video image
    if plot_ground_video:
        plt.figure(figsize=(10,10))
        sns.scatterplot(x = "Valence", y = "Arousal", hue = "Videos", style="Quadrants", 
                        markers=["o", "s", "^", "P"] ,data=groundtruth_df)
        plt.legend(loc="center left", bbox_to_anchor=(1,0.5))
    
    
    #plot ground_label_individual image
    if plot_individual:
        plt.figure(figsize=(10,10))
        label=1
        for i in range(0,len(arousal)-20,20):
            plt.scatter(valence[i:i+20], arousal[i:i+20], label=str(label))
            label += 1
        plt.title("Self Assessment")
        plt.axis("equal")
        plt.xticks(np.arange(1,10,1))
        plt.yticks(np.arange(1,10,1))
        
        plt.xlabel("Valence")
        plt.ylabel("Arousal")
        plt.show()
    
    return groundtruth_df

def GetPseudoLabel_Kmeans(selfassessment_list, plot=False):
    X=list()
    pseudo_label=list()
    pseudo_label_map=dict()
    y_kmeans_pseudo=list()
    for selfassessment in selfassessment_list:
        for instance in selfassessment:
            X.append(instance)
    X=np.asarray(X)
    
    kmeans_clf=KMeans(n_clusters=4)
    kmeans_clf.fit(X)
    y_kmeans = kmeans_clf.predict(X)
    
    for i in range(len(y_kmeans)):
        if X[i][0] > 5 and X[i][1] > 5:
            pseudo_label_map[y_kmeans[i]] = "HAHV"
        elif X[i][0] < 5 and X[i][1] < 5:
            pseudo_label_map[y_kmeans[i]] = "LALV"
        elif X[i][0] < 5 and X[i][1] > 5:
            pseudo_label_map[y_kmeans[i]] = "LAHV"
        elif X[i][0] > 5 and X[i][1] < 5:
            pseudo_label_map[y_kmeans[i]] = "HALV"

    for i in range(len(y_kmeans)):
        if X[i][0] > 5 and X[i][1] > 5:
            pseudo_label.append("HAHV")
        elif X[i][0] < 5 and X[i][1] < 5:
            pseudo_label.append("LALV")
        elif X[i][0] < 5 and X[i][1] > 5:
            pseudo_label.append("LAHV")
        elif X[i][0] > 5 and X[i][1] < 5:
            pseudo_label.append("HALV")
        else:
            pseudo_label.append(pseudo_label_map[y_kmeans[i]])
    
    for label in pseudo_label:
        if label == "LALV":
            y_kmeans_pseudo.append(0)
        elif label == "LAHV":
            y_kmeans_pseudo.append(1)
        elif label == "HALV":
            y_kmeans_pseudo.append(2)
        elif label == "HAHV":
            y_kmeans_pseudo.append(3)
    
    #plot K-means image with pseudo label processing
    if plot:
        plt.figure(figsize=(10,10))
        plt.scatter(X[:, 0], X[:, 1], c=np.asarray(y_kmeans_pseudo), s=50, cmap='viridis')
        centers = kmeans_clf.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        plt.axis("equal")
        plt.xticks(np.arange(1,10,1))
        plt.yticks(np.arange(1,10,1))
        
        plt.xlabel("Valence")
        plt.ylabel("Arousal")
    
    return pseudo_label, y_kmeans_pseudo


def LoadMatData(path):
    mats=[]
    for i in range(40):
        if i+1 < 10:
            dataName="Data_Preprocessed_P0"+str(i+1)    
            src=path+dataName+"/" +dataName+".mat"
            mat=scipy.io.loadmat(src)
            mats.append(mat)
        else:
            dataName="Data_Preprocessed_P"+str(i+1)    
            src=path+dataName+"/" +dataName+".mat"
            mat=scipy.io.loadmat(src)
            mats.append(mat)
    return mats

#organising data
def OrganisingData(mats):        
    data_list=[]
    selfassessment_list=[]   
        
    for mat in mats:
        #short videos only
        data=mat['joined_data'][0][0:16]
        
        #Instaces * Channel -> Channel * Instances
        for i in range(16):
            data[i]=np.transpose(data[i],(1,0))
        data_list.append(data)
        
        #short videos only
        selfassessment=mat['labels_selfassessment'][0][0:16]
        
        #Arousal and Valence
        for i in range(16):
            selfassessment[i] = selfassessment[i][0][0:2]
        selfassessment_list.append(selfassessment)
    return data_list, selfassessment_list


if __name__=='__main__':
    mat_file_path="C:/Users/USER/Desktop/NCTU/Class/2019Autumn/Project/dataset/"
        
    mats=LoadMatData(mat_file_path)
    data_list, selfassessment_list=OrganisingData(mats)
    pseudo_label, pseudo_index=GetPseudoLabel_Kmeans(selfassessment_list, plot=False)
    groundtruth_df=GetGroundtruth_Df(selfassessment_list, pseudo_label, pseudo_index, plot_ground_video=False, plot_individual=True)
    
    
    with open("./tmp/data_list.pkl", "wb") as file:
        pickle.dump(data_list, file)
    groundtruth_df.to_csv("./groundtruth.csv", index=False)
    
