# Keras
import keras
from keras import regularizers
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.utils import np_utils
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from keras import losses, models
from keras.activations import relu, softmax
from keras.layers import (Convolution2D, GlobalAveragePooling2D, BatchNormalization, Flatten, Dropout,
                          GlobalMaxPool2D, MaxPool2D, concatenate, Activation, Input, Dense)
from tensorflow.keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Dropout, Conv2D
# sklearn
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Other  
from tqdm import tqdm, tqdm_pandas
import scipy
from scipy.stats import skew
import pickle
import librosa
import librosa.display
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import pandas as pd
import seaborn as sns
import glob 
import os
import sys
import IPython.display as ipd  # To play sound in the notebook
import warnings
# ignore warnings 
if not sys.warnoptions:
    warnings.simplefilter("ignore")


'''
1. Data Augmentation method   
'''
def speedNpitch(data):
    """
    Speed and Pitch Tuning.
    """
    # you can change low and high here
    length_change = np.random.uniform(low=0.8, high = 1)
    speed_fac = 1.2  / length_change # try changing 1.0 to 2.0 ... =D
    tmp = np.interp(np.arange(0,len(data),speed_fac),np.arange(0,len(data)),data)
    minlen = min(data.shape[0], tmp.shape[0])
    data *= 0
    data[0:minlen] = tmp[0:minlen]
    return data

'''
2. Extracting the MFCC feature as an image (Matrix format).  
'''
def prepare_data(df, n, aug, mfcc):
    X = np.empty(shape=(df.shape[0], n, 216, 1))
    input_length = sampling_rate * audio_duration
    
    cnt = 0
    for fname in tqdm(df.path):
        file_path = fname
        data, _ = librosa.load(file_path, sr=sampling_rate
                               ,res_type="kaiser_fast"
                               ,duration=2.5
                               ,offset=0.5
                              )

        # Random offset / Padding
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length+offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, int(input_length) - len(data) - offset), "constant")

        # Augmentation? 
        if aug == 1:
            data = speedNpitch(data)
        
        # which feature?
        if mfcc == 1:
            # MFCC extraction 
            MFCC = librosa.feature.mfcc(data, sr=sampling_rate, n_mfcc=n_mfcc)
            MFCC = np.expand_dims(MFCC, axis=-1)
            X[cnt,] = MFCC
            
        else:
            # Log-melspectogram
            melspec = librosa.feature.melspectrogram(data, n_mels = n_melspec)   
            logspec = librosa.amplitude_to_db(melspec)
            logspec = np.expand_dims(logspec, axis=-1)
            X[cnt,] = logspec
            
        cnt += 1
    
    return X


'''
3. Confusion matrix plot 
'''        
def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.savefig("confusion_matrix.png")

'''
# 4. Create the 2D CNN model 
'''
def get_2d_conv_model(n):
    ''' Create a standard deep 2D convolutional neural network'''
    nclass = 14
    inp = Input(shape=(n,216,1))  #2D matrix of 30 MFCC bands by 216 audio length.
    x = Convolution2D(32, (4,10), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.2)(x)
    
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.2)(x)
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.2)(x)
    
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.2)(x)
    
    x = Flatten()(x)
    x = Dense(64)(x)
    x = Dropout(rate=0.2)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(rate=0.2)(x)
    
    out = Dense(nclass, activation=softmax)(x)
    model = models.Model(inputs=inp, outputs=out)
    
    opt = optimizers.Adam(0.001)
    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model

# Cosine Loss
import tensorflow as tf
def getCosLoss(predict, label):
    loss = 1 - tf.reduce_mean(tf.losses.cosine_similarity(label, predict))
    return loss

# Cosine Loss
from sklearn.metrics.pairwise import cosine_similarity
#效率高
def getCosAcc(predict, label):
    return tf.reduce_mean(-tf.losses.cosine_similarity(predict, label, axis=0))

def getCosAcc1(predict, label):
    #print(predict[0].shape)
    acc = 0
    sum = 0
    for i in range(predict.shape[0]):
        sum += -tf.losses.cosine_similarity(predict[i], label[i])
    acc = sum / predict.shape[0]
    return acc

# Cosine Loss
import tensorflow as tf
#high efficiency
def getAcc(predict, label):
    #print(predict.shape)
    return -tf.losses.cosine_similarity(predict, label)

def get_2d_conv_auto_encoder(n):
    ''' Create a standard deep 2D convolutional neural network'''
    input_img = Input(shape=(n,216,1))  #2D matrix of 32 MFCC bands by 216 audio length.

    conv_1 = Conv2D(32, (4,10), activation='relu', padding='same')(input_img)
    conv_1 = BatchNormalization()(conv_1)

    pool_1 = MaxPool2D()(conv_1)
    pool_1 = Dropout(rate=0.2)(pool_1)
    
    conv_2 = Conv2D(16, (4,10), activation='relu', padding='same')(pool_1)
    conv_2 = BatchNormalization()(conv_2)

    pool_2 = MaxPool2D()(conv_2)
    pool_2 = Dropout(rate=0.2)(pool_2)

    conv_3 = Conv2D(8, (4,10), activation='relu', padding='same')(pool_2)
    conv_3 = BatchNormalization()(conv_3)

    pool_3 = MaxPool2D()(conv_3)
    pool_3 = Dropout(rate=0.2)(pool_3)

    conv_4 = Conv2D(1, (4,10), activation='relu', padding='same')(pool_3)
    encoded = BatchNormalization(name="encoded")(conv_4)

    up_3 = UpSampling2D()(encoded)
    conv_neg_3 = Conv2D(8, (4,10), activation='relu', padding='same')(up_3)
    conv_neg_3 = BatchNormalization()(conv_neg_3)

    up_4 = UpSampling2D()(conv_neg_3)
    conv_neg_4 = Conv2D(16, (4,10), activation='relu', padding='same')(up_4)
    conv_neg_4 = BatchNormalization()(conv_neg_4)

    up_5 = UpSampling2D()(conv_neg_4)
    conv_neg_5 = Conv2D(32, (4,10), activation='relu', padding='same')(up_5)
    conv_neg_5 = BatchNormalization()(conv_neg_5)

    out = Conv2D(1, (4,10), activation='relu', padding='same')(conv_neg_5)
    out = BatchNormalization()(out)


    model = models.Model(inputs=input_img, outputs=out)
    
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=[getCosAcc])
    return model

'''
# 5. Other functions 
'''
class get_results:
    def __init__(self, model_history, model ,X_test, y_test, labels):
        self.model_history = model_history
        self.model = model
        self.X_test = X_test
        self.y_test = y_test             
        self.labels = labels

    def create_plot(self, model_history):
        '''Check the logloss of both train and validation, make sure they are close and have plateau'''
        fig = plt.figure()
        plt.plot(model_history.history['loss'])
        plt.plot(model_history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.show()
        fig.savefig("loss.png")

    def create_plot_acc(self, model_history):
        '''Check the logloss of both train and validation, make sure they are close and have plateau'''
        fig = plt.figure()
        plt.plot(model_history.history['getCosAcc'])
        plt.plot(model_history.history['val_getCosAcc'])
        plt.title('model acc')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        fig.savefig("acc.png")

    def create_results(self, model):
        '''predict on test set and get accuracy results'''
        opt = optimizers.Adam(0.001)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        score = model.evaluate(X_test, y_test, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

    def confusion_results(self, X_test, y_test, labels, model):
        '''plot confusion matrix results'''
        preds = model.predict(X_test, 
                                 batch_size=16, 
                                 verbose=2)
        preds=preds.argmax(axis=1)
        preds = preds.astype(int).flatten()
        preds = (lb.inverse_transform((preds)))

        actual = y_test.argmax(axis=1)
        actual = actual.astype(int).flatten()
        actual = (lb.inverse_transform((actual)))

        classes = labels
        classes.sort()    

        c = confusion_matrix(actual, preds)
        print_confusion_matrix(c, class_names = classes)
    
    def accuracy_results_gender(self, X_test, y_test, labels, model):
        '''Print out the accuracy score and confusion matrix heat map of the Gender classification results'''
    
        preds = model.predict(X_test, 
                         batch_size=16, 
                         verbose=2)
        preds=preds.argmax(axis=1)
        preds = preds.astype(int).flatten()
        preds = (lb.inverse_transform((preds)))

        actual = y_test.argmax(axis=1)
        actual = actual.astype(int).flatten()
        actual = (lb.inverse_transform((actual)))
        
        # print(accuracy_score(actual, preds))
        
        actual = pd.DataFrame(actual).replace({'female_angry':'female'
                   , 'female_disgust':'female'
                   , 'female_fear':'female'
                   , 'female_happy':'female'
                   , 'female_sad':'female'
                   , 'female_surprise':'female'
                   , 'female_neutral':'female'
                   , 'male_angry':'male'
                   , 'male_fear':'male'
                   , 'male_happy':'male'
                   , 'male_sad':'male'
                   , 'male_surprise':'male'
                   , 'male_neutral':'male'
                   , 'male_disgust':'male'
                  })
        preds = pd.DataFrame(preds).replace({'female_angry':'female'
               , 'female_disgust':'female'
               , 'female_fear':'female'
               , 'female_happy':'female'
               , 'female_sad':'female'
               , 'female_surprise':'female'
               , 'female_neutral':'female'
               , 'male_angry':'male'
               , 'male_fear':'male'
               , 'male_happy':'male'
               , 'male_sad':'male'
               , 'male_surprise':'male'
               , 'male_neutral':'male'
               , 'male_disgust':'male'
              })

        classes = actual.loc[:,0].unique() 
        classes.sort()    

        c = confusion_matrix(actual, preds)
        print(accuracy_score(actual, preds))
        print_confusion_matrix(c, class_names = classes)

ref = pd.read_csv("../dataset/audio_data_path.csv")
ref.head()

sampling_rate=44100
audio_duration=2.5
n_mfcc = 32
mfcc = prepare_data(ref, n = n_mfcc, aug = 0, mfcc = 1)

# Split between train and test 
X_train, X_test, y_train, y_test = train_test_split(mfcc
                                                    , ref.labels
                                                    , test_size=0.25
                                                    , shuffle=True
                                                    , random_state=42
                                                   )


# one hot encode the target 
lb = LabelEncoder()
y_train_old = y_train
y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))

# Pickel the lb object for future use 
filename = 'labels_2D'
outfile = open('./model/'+filename,'wb')
pickle.dump(lb,outfile)
outfile.close()

# Normalization as per the standard NN process
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train = (X_train - mean)/std
X_test = (X_test - mean)/std

print(X_train.shape)
X_train_label = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2], 1)
print(X_train_label.shape)
print(mfcc.shape)
print(y_train_old.shape)


# Build auto encoder model 
model = get_2d_conv_auto_encoder(n=n_mfcc)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               verbose=1,
                               min_lr=0.5e-6)
callbacks = [lr_reducer]
model_history = model.fit(X_train, X_train, validation_data=(X_test, X_test), 
                    batch_size=32, verbose = 1, epochs=100, callbacks=callbacks)

results = get_results(model_history,model,X_test,X_test, ref.labels.unique())
results.create_plot(model_history)
results.create_plot_acc(model_history)

# Save model and weights
model_name = 'model_2D_auto_encoder.h5'
save_dir = "./model/"

model_path = save_dir + model_name
model.save(model_path)
print('Save model and weights at %s ' % model_path)

# Save the model to disk
model_json = model.to_json()
with open("./model/model_json_2D_auto_encoder.json", "w") as json_file:
    json_file.write(model_json)

# loading json and model architecture 
json_file = open('./model/model_json_2D_auto_encoder.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("./model/model_2D_auto_encoder.h5")
print("Loaded model from disk")
 
# Keras optimiser
#opt = optimizers.Adam(0.001)
loaded_model.compile(optimizer="adam", loss="mean_squared_error")

preds = loaded_model.predict(X_train, 
                         batch_size=32, 
                         verbose=1)

print(X_train.shape)
print(preds.shape)
pred = preds[0].reshape(32, 216)
print(pred.shape)
orignal = X_train[0].reshape(32, 216)
print(orignal.shape)

# MFCC
fig = plt.figure(figsize=(20, 15))
plt.subplot(3,1,1)
librosa.display.specshow(pred, x_axis='time')
plt.ylabel('MFCC')
plt.colorbar()
plt.title('MFCC sepc')
fig.savefig("predMFCC.png")

# MFCC
fig = plt.figure(figsize=(20, 15))
plt.subplot(3,1,1)
librosa.display.specshow(orignal, x_axis='time')
plt.ylabel('MFCC')
plt.colorbar()
plt.title('MFCC sepc')
fig.savefig("orignalMFCC.png")





































