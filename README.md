# AMIGOS-EMOTION-RECOGNITION

In this study, we developed an emotion recognition system based on the valence-arousal model. Independent component analysis (ICA) was applied in order to remove the ocular movement effect. Afterward, we applied discrete wavelet transform (DWT) on the processed EEG signals which was separated to gamma, beta, alpha and theta bands. Shannon’s entropy and signal energy were computed with a temporal window from these 4 channels. Deep convolutional neural network (CNN) model is trained to classify the signal into valence-arousal space.

For more details please checkout **Emotion Recognition based on EEG signals using Deep CNN model.pdf**
## Package Version
---
    python == 3.7.4
    numpy == 1.16.5
    pandas == 0.25.1
    pytorch == 1.0.0
    matplotlib ==3.1.2
    pickle == 4.0
    mne == 0.19.2
    scipy == 1.3.1
    sklearn == 0.20.0
    pywt == 1.0.3


## Execution Process
---
### For more details, check out the **src** block.
1. Run **Readmat.py** to load matlab file from AMIGOS datset.
2. Whether applying ICA for removing ocular movement effect from EEG data or not? If no, execute the ProcessData function in **ICA.py** to get NaN dropped EEG data list. If yes, just run **ICA.py**
3. Run **Preprocessing.py**
4. Set train_encoder=*True*, and train_classifier=*False* in **Train.py** to train **convolutional autoencoder** for **pretraining** while set train_encoder=*False*, and train_classifier=*Train* to train **classifier**


## dataset
---
- We put our AMIGOS dataset in this directory


## src
---
- Containing all **source code** written in python
    ### Readmat.py
    - GetGroundtruth_Df(selfassessment_list, pseudo_label, pseudo_index, plot_ground_video=False, plot_individual=False)
        - Generate ground truth dataframe with labels
        - selfassessment_list(list): contains all self assessment data
        - pseudo_label(list): contains emotion states in arousal-valence style, i.e. HALV, for each instance
        - pseudo_index(list): contains emotion states in tokenized style
        - plot_ground_video(bool): plot a scatter graph in valence-arousal space with legends of video types
        - plot_ground(bool): plot a scatter graph in valence-arousal space with legends of participants
        - return type: dataframe
    - GetPseudoLabel_Kmeans(selfassessment_list, plot=False)
        - Get emotion state labels through K-means
        - selfassessment_list(list) contains all self assessment data
        - plot(bool): plot k-Means clusters in valence-arousal space
        - return: pseudo_label(list), y_kmeans_pseudo(list)(pseudo_index)
    - LoadMatData(path)
        - Load matlab data
        - path(str): the path you put your dataset data
        - return type: list
    - OrganisingData(mats)
        - Extract needed information from dataset data
        - mats(list): dataset data
        - return: data_list, selfassessment_list
    ### ICA.py
    - PreprocessData()
        - Drop EEG data in first three seconds, and remove all NaN data and labels
        - return type: list
    - ICA(EEG_list, index)
        - Perform ocular movement effect removing process with ICA, and dump the processed data in src/eeg_ica/
        - EEG_list(list): a list contains EEG data
        - index(int): the index of EEG data in EEG_list you want to start the ICA process
    - LoadICAData()
        - Load all processed data from src/eeg_ica/ and formed into a list. Afterwards, the function dump the list into src/tmp/ 
    ### Preprocessing.py
    - LoadProcessedData()
        - Load data dumped into src/tmp (formed ICA processed EEG data)
        - return type: list
    - Energy(signal)
        - Compute signal energy
        - signal(numpy array)
        - return type: numpy array
    - DiscreteWaveletTransform(signal, feature, windowsize=4, sampling_rate=128)
        - signal: 2D EEG signal
        - feature(str): "entropy" or "energy"
        - widowsize: the size of the temporal window, default = 4 seconds
        - sampling_rate: the sample rate of the input signal
        - return type: numpy array contaning signals in gamma, alpha, beta, theta frequency bands
    - ZeroPadding(EEG_list, total_length)
        - Perform zero padding
        - EEG_list(list): a list containing all EEG signals you want to pad
        - total_length: The signal length after padding
        - return type: list
    ### Train.py
    - LoadTrainTestData()
        - Load all EEG data and labels, and split training and testing data
        - return type: numpy array, numpy array, numpy array, numpy array
    - TrainAutoEncoder(train_loader, test_loader, model, epochs, device, optimizer='Adam', lr=1e-4):
        - train the model and save the model weight dictionary 
        - train_loader: training dataloader(torch dataloader)
        - test_loader: testing dataloader(torch dataloader)
        - model(torch model): the model you want to optimized
        - epochs(int): how many epochs you want to train
        - device: which gpu or cpu device you wnat to use
        - optimizer: which optimization algorithm you want to use (only Adam is avaliable now)
        - lr: learning rate
        - return: training loss(list), testing loss(list)
    - TestAutoEncoder(test_loader, model, loss_fct)
        - test_loader: testing dataloader(torch dataloader)
        - model: model you want to validate
        - loss_fct: loss function for validation
        - return type: average testing loss
    - TrainClassifier(train_loader, test_loader, model, epochs, device, optimizer='Adam', lr=1e-4):
        - train the model and save the model weight dictionary 
        - train_loader: training dataloader(torch dataloader)
        - test_loader: testing dataloader(torch dataloader)
        - model(torch model): the model you want to optimized
        - epochs(int): how many epochs you want to train
        - device: which gpu or cpu device you wnat to use
        - optimizer: which optimization algorithm you want to use (only Adam is avaliable now)
        - lr: learning rate
        - return: training loss, testing loss, training accuracy, testing accuracy, valence accuracy, arousal accuracy
    - TestClassifier(test_loader, model, loss_fct)
        - test_loader: testing dataloader(torch dataloader)
        - model: model you want to validate
        - loss_fct: loss function for validation
        - return type: average testing loss, average testing accuracy, average valence accuracy, average arousal accuracy

    

## src/modelweight
---
- Containing all **model weight** (torch model state dictionary object)


## src/tmp
---
- Containing all **processed data**

## src/eeg_ica
---
- Containing all **EEG data** processed by **ICA**

