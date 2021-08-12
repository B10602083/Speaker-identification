# -*- coding: utf-8 -*-
"""Speaker.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tIm0f79kX7MPvJTmlV--osOZLq7p0Jf1
"""

!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse
from google.colab import auth
auth.authenticate_user()
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}

!mkdir -p drive
!google-drive-ocamlfuse drive

import os
os.chdir("/content/drive/voxforge")

# Commented out IPython magic to ensure Python compatibility.
import os
from pathlib import Path

# %matplotlib inline
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt
import librosa
import librosa.display

"""

```
# 此內容會顯示為程式碼
```

"""

def fetch_voxforge_data():
    ''' Fetch tar archive hosted from s3 bucket and untar it. '''
    root_url = 'http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/'
    archive_tar = 'voxforge.tar.gz'
    data_archive = keras.utils.get_file(archive_tar, root_url + archive_tar, extract=True)
    return data_archive

def is_valid(file_path):
    ''' returns True if a regular files. False for hidden files.
    Also, True is a known user with a name, False if anon.
    '''
    file_name = tf.strings.split(file_path, '/')[-1]
    if tf.strings.substr(file_name, 0, 1) == tf.constant(b'.'):
        return False
    sc = tf.strings.split(file_path, '/')[-3]
    speaker = tf.strings.split(sc, '-')[0]
    return not tf.strings.substr(speaker, 0, 9) == tf.constant(b'anonymous')

list_ds = tf.data.Dataset.list_files(str("/content/drive/voxforge/"'*/wav/*.wav'))
list_ds = list_ds.filter(is_valid)
for f in list_ds.take(3):
  print(f.numpy())

def extract_speaker(file_path):
    ''' extract speaker name from the file path '''
    sc = tf.strings.split(file_path, '/')[-3]
    return tf.strings.split(sc, '-')[0]

speaker_ds = list_ds.map(extract_speaker)
for speaker in speaker_ds.take(50):
    print(speaker)

speaker_encoder = preprocessing.LabelEncoder()
speaker_idx = speaker_encoder.fit_transform([bytes.decode(s.numpy()) for s in speaker_ds])
encoded_speaker_ds = tf.data.Dataset.from_tensor_slices(speaker_idx)
unique_speakers = len(speaker_encoder.classes_)
for es in encoded_speaker_ds.take(50):
    print(es)

sample_audio = os.path.join("/content/drive/voxforge/", '/content/drive/voxforge/Campbell-20091230-set/wav/a0583.wav')
import IPython.display as ipd
ipd.Audio(sample_audio)

x, sr = librosa.load(sample_audio)
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)

X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')

def wav2mfcc(file_path, max_pad_len=196):
    ''' convert wav file to mfcc matrix with truncation and padding '''
    wave, sample_rate = librosa.load(file_path, mono=True, sr=None)
    mfcc = librosa.feature.mfcc(wave, sample_rate)
    mfcc = mfcc[:, :max_pad_len]
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc

def extract_mfcc(file_path):
    ''' returns 3D tensor of the mfcc coding from the wav file '''
    file_name = bytes.decode(file_path.numpy())
    mfcc = tf.convert_to_tensor(wav2mfcc(file_name))
    mfcc = tf.expand_dims(mfcc, 2)
    return mfcc

def create_audio_ds(list_ds):
    ''' creates audio dataset containing audio tensors from file list dataset '''
    batch = []
    for f in list_ds:
        audio = extract_mfcc(f)
        batch.append(audio)
    return tf.data.Dataset.from_tensor_slices(batch)

# Commented out IPython magic to ensure Python compatibility.
# %time audio_ds = create_audio_ds(list_ds)

for a in audio_ds.take(10):
    print(a.numpy().shape)

complete_labeled_ds = tf.data.Dataset.zip((audio_ds, encoded_speaker_ds))

input_shape = None
for audio, speaker in complete_labeled_ds.take(1):
    input_shape = audio.shape
    print('input_shape', audio.shape)
    print('output_shape', speaker.shape)

labeled_ds = complete_labeled_ds

data_size = sum([1 for _ in labeled_ds])
train_size = int(data_size * 0.9)
val_size = int(data_size * 0.05)
test_size = data_size - train_size - val_size
print('all samples: {}'.format(data_size))
print('training samples: {}'.format(train_size))
print('validation samples: {}'.format(val_size))
print('test samples: {}'.format(test_size))

batch_size = 32
labeled_ds = labeled_ds.shuffle(data_size, seed=42)
train_ds = labeled_ds.take(train_size).shuffle(1000).batch(batch_size).prefetch(1)
val_ds = labeled_ds.skip(train_size).take(val_size).batch(batch_size).prefetch(1)
test_ds = labeled_ds.skip(train_size + val_size).take(test_size).batch(batch_size).prefetch(1)

def create_model():
    dropout_rate = .25
    regularazation = 0.001
    audio_input = keras.layers.Input(shape=input_shape)
    conv1 = keras.layers.Conv2D(16, kernel_size=(3, 3), padding='same',
                               activation='relu', input_shape=input_shape)(audio_input)
    maxpool1 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(conv1)
    batch1 = keras.layers.BatchNormalization()(maxpool1)
    conv2 = keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same',
                               activation='relu', input_shape=input_shape)(batch1)
    maxpool2 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)
    batch2 = keras.layers.BatchNormalization()(maxpool2)
    conv3 = keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', 
                activation='relu')(batch2)
    maxpool3 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(conv3)
    batch3 = keras.layers.BatchNormalization()(maxpool3)
    flt = keras.layers.Flatten()(batch3)
    drp1 = keras.layers.Dropout(dropout_rate)(flt)
    dense1 = keras.layers.Dense(unique_speakers * 2, activation='relu',
                kernel_regularizer=keras.regularizers.l2(regularazation))(drp1)
    drp2 = keras.layers.Dropout(dropout_rate)(dense1)
    output = keras.layers.Dense(unique_speakers, activation='softmax', name='speaker')(drp2)
    model = keras.Model(inputs=audio_input, outputs=output)
    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['acc'])
    return model

train_model = False
model_name = 'spr_model.h5'
model_path = os.path.join('.', model_name)
model = None
if os.path.exists(model_path):
    model = keras.models.load_model(model_path)
else:
    model = create_model()
    train_model = True

model.summary()

if train_model:
    root_logdir = os.path.join(os.curdir, "spr_logs")
    def get_run_dir():
        import time
        run_id = time.strftime("run%Y_%m_%d-%H_%M_%S")
        return os.path.join(root_logdir, run_id)
    run_logdir = get_run_dir()
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir, update_freq='batch')
    history = model.fit(train_ds, epochs=70, validation_data=val_ds, callbacks=[tensorboard_cb])

# In[] plt
plt.figure(figsize = (15,5))
plt.subplot(1,2,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.figure(figsize = (15,5))
plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.evaluate(test_ds)

if train_model:
    model.save(model_name)

sample_file = [os.path.join('/content/drive/voxforge/Aaron-20080318-kdl/wav/b0019.wav'),
               os.path.join('/content/drive/voxforge/bugsysservant-20091103-cob/wav/b0078.wav'),
               os.path.join('/content/drive/voxforge/Campbell-20091230-set/wav/a0583.wav'),
               os.path.join('/content/drive/voxforge/DavidL-20091116-kth/wav/b0056.wav'),
               os.path.join('/content/drive/voxforge/ESimpray-20150125-svl/wav/b0025.wav'),
               os.path.join('/content/drive/voxforge/Fandark-20100822-acy/wav/b0003.wav'),
               os.path.join('/content/drive/voxforge/GamaBedolla-20150210-jbr/wav/b0404.wav'),
               os.path.join('/content/drive/voxforge/Hadlington-20130720-pwc/wav/a0210.wav'),
               os.path.join('/content/drive/voxforge/J0hnny_b14z3-20111219-ibu/wav/b0051.wav'),
               os.path.join('/content/drive/voxforge/Kai-20111021-apo/wav/b0049.wav'),
               os.path.join('/content/drive/voxforge/L1ttl3J1m-20090701-fhz/wav/a0185.wav'),
               os.path.join('/content/drive/voxforge/MARTIN0AMY-20111106-pwg/wav/a0491.wav'),
               os.path.join('/content/drive/voxforge/Nadim-20100515-efk/wav/b0276.wav'),
               os.path.join('/content/drive/voxforge/Otuyelu-20101107-crp/wav/b0209.wav'),
               os.path.join('/content/drive/voxforge/Paddy-20100120-msy/wav/b0092.wav')]
sample_ds = tf.data.Dataset.from_tensor_slices(sample_file)
sample_input = create_audio_ds(sample_ds).batch(2)
output = model.predict(sample_input)

speaker_ids = output.argmax(axis=1)
speakers = speaker_encoder.inverse_transform(speaker_ids)
print(speakers)
print(output)



from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion Matrix", cmap=plt.cm.Blues):
   plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
   plt.rcParams['axes.unicode_minus'] = False
   classes_name = ['Aaron', 'bugsysservant', 'Campbell', 'DavidL', 'ESimpray', 'Fandark', 'GamaBedolla', 'Hadlington', 'J0hnny_b14z3', 'Kai', 'L1ttl3J1m', 'MARTIN0AMY', 'Nadim', 'Otuyelu', 'Paddy']
   plt.figure()
   plt.imshow(cm, interpolation='nearest', cmap=cmap)
   plt.title(title)
   plt.colorbar()
   tick_marks = np.arange(len(classes_name))
   plt.xticks(tick_marks, classes_name, rotation=45)
   plt.yticks(tick_marks, classes_name)
   if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
   thresh = cm.max() / 2
   for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

   plt.tight_layout()
   plt.ylabel('True label')
   plt.xlabel('Predicted label')
   plt.savefig('Confusion Matrix.png')

labels = ['Aaron', 'bugsysservant', 'Campbell', 'DavidL', 'ESimpray', 'Fandark', 'GamaBedolla', 'Hadlington', 'J0hnny_b14z3', 'Kai', 'L1ttl3J1m', 'MARTIN0AMY', 'Nadim', 'Otuyelu', 'Paddy']
labels = np.array(labels)
results = model.predict(test_ds) # X_test是測試資料
cm = confusion_matrix(labels, speakers) # labels為分類名稱的array，results是測試結果
plot_confusion_matrix(cm, range(0, 5)) # range(0, 5)表示有5個分類