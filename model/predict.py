import numpy as np
import tensorflow as tf
import pandas as pd
import os
import datetime
import math
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import preprocess_input
from keras.metrics import mean_absolute_error
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense,Flatten
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import Xception
import time
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error
import csv

def Prediciendo(nombre, ba, male, path):
    imgSize = 256
    media = 127.3207517246848
    desviacionS = 41.182021399396326


    print("Ingrese")
    #nombre = input() #'Hueso2.png'
    #ba = float(input()) #168
    #male = input() #False
    #path = input() #'./wasd'

    df = pd.DataFrame(columns=['id', 'boneage', 'male'])
    df.loc[len(df.index)] = [nombre] + [float(ba)] + [male]
    df['val_z'] = (df['boneage'] - media) / desviacionS
    df['gender'] = df['male'].apply(lambda x: 'male' if x else 'female')

    validationDataGenerator = ImageDataGenerator(preprocessing_function = preprocess_input)


    testX, testY = next(validationDataGenerator.flow_from_dataframe(dataframe = df, directory = path, x_col = 'id', y_col = 'val_z', target_size = (imgSize, imgSize), batch_size = 2523, class_mode = 'raw'))

    def mae(xP, yP):
        return mean_absolute_error((desviacionS * xP + media), (desviacionS * yP + media))


    #Modelo
    modelo1 = Xception(input_shape = (imgSize, imgSize, 3), include_top = False, weights = 'imagenet')
    modelo1.trainable = True

    modelo2 = Sequential()
    modelo2.add(modelo1)
    modelo2.add(GlobalMaxPooling2D())
    modelo2.add(Flatten())
    modelo2.add(Dense(10, activation = 'relu'))
    modelo2.add(Dense(1, activation = 'linear'))
    modelo2.compile(loss = 'mse', optimizer = 'adam', metrics = [mae])

    modelo2.load_weights('modelo.h5')

    start_time = time.time()

    prediccion = media + desviacionS * (modelo2.predict(testX, batch_size = 32, verbose = True))


    print(prediccion) #aqui es donde debe de ir len(df.index)

    #Mostrando imagen
    fig, axS = plt.subplots()
    axS.imshow(testX[len(testX) - 1], cmap = 'bone')
    axS.set_title('Edad: %f\nPrediccion: %f ---- En %f segundos' % (float(ba)/12.0, prediccion[len(prediccion) - 1] /12.0, (time.time() - start_time)))
    axS.axis('off')

    plt.rcParams['figure.dpi'] = 300

    plt.show()

    #fig.savefig(('ImagenResultante.png'), dpi = 300)
