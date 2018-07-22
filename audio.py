# coding= UTF-8
#
# Author: zakordonets
# Date  : 21.07.2018
#
import glob
import os
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn import preprocessing

# Exptracting Features from sounds
def extract_feature(file_name):
    X, sample_rate = sf.read(file_name, dtype='float32')
    if X.ndim > 1:
        X = X[:,0]
    X = X.T

    # short term fourier transform
    stft = np.abs(librosa.stft(X))

    # mfcc
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

    # chroma
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

    # melspectrogram
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)

    # spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    
    return mfccs,chroma,mel,contrast,tonnetz

# Loading train files
def t_parse_audio_files(parent_dir, desc_table):
    features, labels = np.empty((0,193)), np.empty(0)
    
    for index, row in tqdm(desc_table.iterrows(), 
                           total = desc_table.shape[0]):
        fn = os.path.join(parent_dir, row['x0'])
        try:
            mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
        except Exception as e:
            print("[Error] extract feature error. %s" % (e))
            continue
        ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
        features = np.vstack([features,ext_features])
        labels = np.append(labels, row['x4'])
    return np.array(features), np.array(labels)

# Loading test files
def f_parse_audio_files(parent_dir,file_ext='*.wav'):
     features, files = np.empty((0,193)), np.empty(0)
     for fn in tqdm(glob.glob(os.path.join(parent_dir, file_ext))):
         try:
             mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
         except Exception as e:
             print("[Error] extract feature error. %s" % (e))
             continue
         ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
         features = np.vstack([features,ext_features])
         
         files = np.append(files, os.path.split(fn)[1])

     return np.array(features), files



# Get features and labels
# Loading meta file with list of files and classes
meta = pd.read_csv("e:\Datasets\data_v_7_stc\meta\meta.txt", sep = "\t",
                   header = None, prefix = 'x')
features, labels = t_parse_audio_files('E:\Datasets\data_v_7_stc\\audio', meta)

# Encoding Labels
le = preprocessing.LabelEncoder()
le_lab = le.fit_transform(labels)

# Save loaded and decoded features an labels 
np.save('feat.npy', features)
np.save('label.npy', le_lab)


# Prepare the data
X = np.load('feat.npy')
y = np.load('label.npy').ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=233)

# Build the Convolutional Neural Network
model = Sequential()

model.add(Conv1D(64, 3, activation='relu', input_shape=(193, 1)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(units=64, activation='relu', input_dim=128))
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# Convert label to onehot
y_train = keras.utils.to_categorical(y_train, num_classes=8)
y_test = keras.utils.to_categorical(y_test, num_classes=8)

X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

model.fit(X_train, y_train, batch_size=64, epochs=30)
score, acc = model.evaluate(X_test, y_test, batch_size=16)

print('Test score:', score)
print('Test accuracy:', acc)

# Loading test files
os.path.split(glob.glob(os.path.join('E:\Datasets\\data_v_7_stc\\test', '*.wav'))[0])[1]
test_features, filenames = f_parse_audio_files('E:\Datasets\\data_v_7_stc\\test')

# predicting results
test_features = np.expand_dims(test_features, axis = 2)
results = model.predict(test_features)

# Convert results to output form
num_results = np.empty((0,2))
for i in results:
    res_str = np.hstack([i.max(), np.argmax(i)])
    num_results = np.vstack([num_results, res_str])
   
exit_res = pd.DataFrame({ 'Filenames' : filenames,
                         'Score' : num_results[:,0],
                         'Class' : num_results[:,1]})

exit_res['Class'] = le.inverse_transform(exit_res['Class'].astype(int))

#Save results to disk
exit_res.to_csv("e:\Datasets\data_v_7_stc\output.txt", sep = "\t", 
                header = None, index = False, float_format = "%.3f")