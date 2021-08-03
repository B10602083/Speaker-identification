import h5py
import numpy as np
from sklearn import preprocessing
import os
from pathlib import Path
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import glob
#import plot_confusion_matrix as pt
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog as fd
import os
from PIL import Image, ImageTk

new_model = ''
labels = ''
file_list = []

def read_h5file(model_path = 'spr_model.h5'):
    # f = h5py.File('path/filename.h5','r') #打开h5文件
    # f = h5py.File('C:/Users/kaka5/OneDrive/Desktop/spr_model.h5','r')
    # model_path = r"C:/Users/kaka5/OneDrive/Desktop/spr_model.h5"
    f = h5py.File(model_path,'r')
    f.keys() #可以查看所有的主键
    print([key for key in f.keys()])
    
    # Recreate the exact same model, including its weights and the optimizer
    # new_model = tf.keras.models.load_model('C:/Users/kaka5/OneDrive/Desktop/spr_model.h5')
    global new_model
    new_model = tf.keras.models.load_model(model_path)
    
    # Show the model architecture
    new_model.summary()
    
    return new_model

def extract_speaker(file_path):
    ''' extract speaker name from the file path '''
    sc = tf.strings.split(file_path, '\\')[-3]
    return tf.strings.split(sc, '-')[0]

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
    batch = []
    for f in list_ds:
        audio = extract_mfcc(f)
        batch.append(audio)
    return tf.data.Dataset.from_tensor_slices(batch)

def audio_predict(sample_file, new_model, label_file = 'label.txt'):
    global labels
    labels = np.loadtxt(label_file, str, delimiter='\t')
        
    sample_ds = tf.data.Dataset.from_tensor_slices(sample_file)
    sample_input = create_audio_ds(sample_ds).batch(2)
    
    output = new_model.predict(sample_input)
    
    print(output)
    
    speaker_ids = output.argmax(axis=1)
    print(speaker_ids)
    #speakers = speaker_encoder.inverse_transform(speaker_ids) 
    #speakers = speaker_encoder.inverse_transform([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
    speakers = labels[speaker_ids]
    print(speakers)
    print(output)
    return speakers

def show(sample_audio):
    x, sr = librosa.load(sample_audio)
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(x, sr=sr)
    plt.savefig('spec.png')
    plt.close()
    


#html = PhotoImage(file="test123.png")
#Label(root,image=html).pack()

def browsefunc():
    global labels
    global new_model
    global file_list
    new_model = ''
    labels = ''
    file_list = []
    filename = fd.askopenfilename()
    pathlabel.config(text=filename)
    new_model = read_h5file() # 同步讀取h5
    file_list.append(filename)
    #a.config(text=filename)
    
def model_predict():
    # 
    output = audio_predict(file_list, new_model) # 同步讀取h5
    b.config(text=str(output[0]))
    
    # 
    show(file_list[0])
    image1 = Image.open("spec.png")
    image1 = image1.resize((500, 200), Image.ANTIALIAS) # 重設圖片大小
    image1 = ImageTk.PhotoImage(image1) # 加載圖檔
    a.config(image= image1, width=500, height=180) # set image
    a.image = image1 # keep a reference
    
    #a.config(text=filename)
  

 
root = Tk()

root.iconbitmap('Speaker.ico') # 更改圖示

root.title('Speaker')
root.geometry('700x500')


      
l = tk.Label(root,text ='Load wav file', font=('Arial', 12), width=30)
l.pack()

pathlabel = Label(root)
pathlabel.pack()

# 載入檔案
browsebutton = Button(root, text="Browse", command=browsefunc)
browsebutton.pack()

a = tk.Label(root,borderwidth = 5,
         relief="sunken", bg = 'white', text ='顯示載入音檔\n的wavform ',font=('Arial', 12), width=55, height=10)
a.pack(side='left')

b = tk.Label(root,borderwidth = 5,
         relief="sunken", bg = 'white',text ='顯示辨識後\n的結果',font=('Arial', 12), width=20, height=10)
b.pack(side='right')

Button1 = tk.Button(text = "Begin", width=50, command=model_predict).place(x=165, y=400)

Quit = tk.Button( text="QUIT", font = ("Times New Roman", 13, 'bold'), command=root.destroy).place(x=320, y=445)

#Quit.pack(side="bottom")
#Button1.pack(side=tk.BOTTOM)
#.place(x=20, y=250)
root.mainloop()