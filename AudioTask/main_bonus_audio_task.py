# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import librosa
import os
import glob
import math
import librosa.display
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split


from tensorflow.keras import utils as np_utils
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam
from keras import backend as K
from sklearn.model_selection import KFold


import models as models



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


####### Learning rate scheduler ####################
def step_decay(epoch):
    if(epoch < 20):
        lr = 0.001 #0.01
    elif(epoch < 50):
        lr = 0.001
    else:
        lr = 0.0001
    return lr
lrate = LearningRateScheduler(step_decay)


def get_audio_segments(X,y,winsize,overlap,chans,fs):
    
    msec = 1000
    win_samples = int((winsize/msec)*fs)
    # converting the overlap of windows from ms to number of samples
    overlap_samples = int((overlap/msec)*fs)
    
    start = 0
    end   = win_samples
    stride = win_samples - overlap_samples
    
    segments = np.empty((0,chans,win_samples))
    y_segments = np.empty((0,1))
    
    signal_length = X.shape[2]
    NO_segments = math.ceil((signal_length - win_samples)/stride + 1)
    padding     = (NO_segments - 1)*stride - (signal_length -win_samples)
    
    data = np.concatenate((X, np.zeros((X.shape[0],chans,padding))),axis=2)
    
    vec = np.arange(0, signal_length, stride)
    for ind,j in enumerate(vec): 
        if ind <= NO_segments - 1:
            print('the segment number is {}'.format(ind))
            print('the sample number is {}'.format(j))
            
            segment = data[:,:,j: j + win_samples]
            segments = np.concatenate((segments,segment),axis=0)
            y_segments = np.concatenate((y_segments,y)) 
        
    return segments,y_segments


fs      = 22050#Hz
msec    = 1000 # 
chans   = 1
winsize = 3000 #
stride  = 3000 #milisecond
overlap = winsize - stride  #milisecond


audio_path = 'genres_original/'
genres = os.listdir(audio_path)

#y,sr = librosa.load('genres_original/jazz/jazz.00055.wav')
data = np.zeros(((len(genres[0:4])*100),617400))
label = np.zeros(((len(genres[0:4])*100),1))


audio_ind = 0
for genre_ind, genre in enumerate(genres[0:4]):
    genre_path = audio_path + genre + '/*.wav'
    audio_files = glob.glob(genre_path)
    for file in audio_files:
        y,sr = librosa.load(file) 
        audio_file, _ = librosa.effects.trim(y)
        data[audio_ind,:] = audio_file[0:617400]
        label[audio_ind,:] = genre_ind
        audio_ind += 1
        

min_max_scaler = preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(data)
data = np.expand_dims(data,axis=1)

data, labels = get_audio_segments(data,label,winsize=winsize,overlap=overlap,chans=chans,fs=fs)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

# expand dims to match the requirements of network architecture 
X_train = np.expand_dims(X_train,axis=-1)
X_test = np.expand_dims(X_test,axis=-1)  

y_train = np_utils.to_categorical(y_train)
y_test  = np_utils.to_categorical(y_test)



########### model parameters initialization #################

n_epochs           = 100 
# model settings 
kernLength         = 250
poolLength         = 10 
num_splits         = 9
acc                = np.zeros((num_splits,2))
batch_size         = 350


n_samples = np.shape(X_test)[2]
model = models.EEGNet(nb_classes=4, Chans=chans, Samples=n_samples, regRate=.25,
			   dropoutRate=0.1, kernLength=250,poolLength=8, 
			   numFilters=8, dropoutType='Dropout')


# Set Learning Rate
adam_alpha = Adam(lr=(0.0001)) #lr=(0.0001)
model.compile(loss='categorical_crossentropy', optimizer=adam_alpha, metrics = ['accuracy'])


history = model.fit(X_train, y_train, 
        validation_data=(X_test, y_test),
        batch_size = batch_size, epochs = n_epochs, callbacks=[lrate], verbose = 2)

# metrics  
results = np.zeros((4,len(history.history['accuracy'])))
results[0] = history.history['accuracy']
results[1] = history.history['val_accuracy']
results[2] = history.history['loss']
results[3] = history.history['val_loss']

acc = results[0:2,-1]

print('accuracy {:}\t{:.4f}'.format(acc[0], acc[1]))







    
    