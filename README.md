# AMIGOS-EMOTION-RECOGNITION

In this study, we developed an emotion recognition system based on the valence-arousal model. Independent component analysis (ICA) was applied in order to remove the ocular movement effect. Afterward, we applied discrete wavelet transform (DWT) on the processed EEG signals which was separated to gamma, beta, alpha and theta bands. Shannon’s entropy and signal energy were computed with a temporal window from these 4 channels. Deep convolutional neural network (CNN) model is trained to classify the signal into valence-arousal space.

For more details please checkout **Emotion Recognition based on EEG signals using Deep CNN model.pdf**
- ### Package Version
---
    numpy == 1.16.5
    pandas == 0.25.1
    pytorch == 1.0.0
    matplotlib ==3.1.2
    pickle == 4.0
    mne == 0.19.2
    scipy == 1.3.1
    sklearn == 0.20.0
    pywt == 1.0.3


- ## Execution Process
---
### For more details, check out the **src** block.
1. Run **Readmat.py** to load matlab file from AMIGOS datset. 
2. Whether applying ICA for removing ocular movement effect from EEG data or not? If no, execute the ProcessData function in **ICA.py** to get NaN dropped EEG data list. If yes, just run **ICA.py**
3. Run **Preprocessing.py**
4. Set train_encoder=*True*, and train_classifier=*False* in **Train.py** to train **convolutional autoencoder** for **pretraining** while set train_encoder=*False*, and train_classifier=*Train* to train **classifier**


- ## dataset
---
- We put our AMIGOS dataset in this directory


- ## src
---
- Containing all **source code** written in python
    ###Readmat.py

- ## src/modelweight
---
- Containing all **model weight** (torch model state dictionary object)
- 

- ## src/tmp
---
- Containing all **processed data**

- ## src/eeg_ica
---
- Containing all **EEG data** processed by **ICA**
