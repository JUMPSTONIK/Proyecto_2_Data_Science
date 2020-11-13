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
from tensorflow.keras.applications.xception import Xception
import time
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error

#directorios
training = './archive/boneage-training-dataset/boneage-training-dataset'
testing = './archive/boneage-test-dataset/boneage-test-dataset'

#Train y test
train = pd.read_csv('boneage-training-dataset.csv')
test =  pd.read_csv('boneage-test-dataset.csv')

train['id'] = train['id'].apply(lambda x: str(x)+'.png')
test['Case ID'] = test['Case ID'].apply(lambda x: str(x)+'.png')

print(train.head())

train['gender'] = train['male'].apply(lambda x: 'male' if x else 'female')


print(train['gender'].value_counts())
sb.countplot(x = train['gender'])

#Normalizando datos para valor z
media = train['boneage'].mean()
desviacionS = train['boneage'].std()

train['val_z'] = (train['boneage'] - media) / desviacionS

print(train.head())

#Separando dataframe
df_train, df_val = train_test_split(train, test_size = 0.25, random_state = 0)

imgSize = 256

#Data Generators
trainDataGenerator = ImageDataGenerator(preprocessing_function = preprocess_input)
validationDataGenerator = ImageDataGenerator(preprocessing_function = preprocess_input)


trainGen = trainDataGenerator.flow_from_dataframe(dataframe = df_train, directory = training, x_col= 'id', y_col= 'val_z', batch_size = 32, seed = 42, shuffle = True, class_mode= 'raw', flip_vertical = True, color_mode = 'rgb', target_size = (imgSize, imgSize))
valGen = validationDataGenerator.flow_from_dataframe(dataframe = df_val, directory = training, x_col = 'id', y_col = 'val_z', batch_size = 32, seed = 42, shuffle = True, class_mode = 'raw', flip_vertical = True, color_mode = 'rgb', target_size = (imgSize, imgSize))


testDataGenerator = ImageDataGenerator(preprocessing_function = preprocess_input)
testGen = testDataGenerator.flow_from_directory(directory = testing, shuffle = True, class_mode = None, color_mode = 'rgb', target_size = (imgSize, imgSize))

testX, testY = next(validationDataGenerator.flow_from_dataframe(dataframe = df_val, directory = training, x_col = 'id', y_col = 'val_z', target_size = (imgSize, imgSize), batch_size = 2523, class_mode = 'raw'))

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

print(modelo2.summary())

#Guardande el modelo
earlyStop = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 5, verbose = 0, mode = 'auto')
modelCheckpoint = ModelCheckpoint('modelo.h5', monitor = 'val_loss', mode = 'min', save_best_only = True)
#, datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
direccionLog = os.path.join('/model', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
modelo2.save('./model/modelo.h5')
print(direccionLog)
cb = TensorBoard(direccionLog, histogram_freq = 1)

reducir_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 10, verbose = 0, mode = 'auto', min_delta = 0.0001, cooldown = 0, min_lr = 0)
callbacks = [cb, earlyStop, modelCheckpoint, reducir_lr]


#Prediciendo el modelo
#fitModel = modelo2.fit(trainGen, steps_per_epoch = 315, validation_data = valGen, validation_steps = 1, epochs = 50)

start_time = time.time()

modelo2.load_weights('modelo.h5')

prediccion = media + desviacionS * (modelo2.predict(testX, batch_size = 32, verbose = True))
testMeses = media + desviacionS * testY

ordInd = np.argsort(testY)
ordInd = ordInd[np.linspace(0, len(ordInd)- 1, 8).astype(int)]

print(ordInd, "----------------")



ind = []
num = int(abs((len(testX) - 1) * 0.66))

for i in range(num):
    ind.append(random.randint(0, len(testX) - 1))



print("Error medio absoluto: ", mean_absolute_error(testMeses, prediccion))

#Mostrando cada imagen
for i in range(len(ind)):
    fig, axS = plt.subplots()
    axS.imshow(testX[ind[i], :, :, 0], cmap = 'bone')
    axS.set_title('Edad: %f\nPrediccion: %f' % (testMeses[ind[i]]/12.0, prediccion[ind[i]]/12.0))
    axS.axis('off')
        

    print((time.time() - start_time), " segundos")

    fig.savefig(('prediccion_imagen'+str(i)+'.png'), dpi = 300)

#Accuracy y grafica de regresion
print("accuracy: ", mean_squared_error(testMeses, prediccion))
plt.scatter(testMeses, prediccion)

m, b = np.polyfit(testMeses, prediccion, 1)
plt.plot(testMeses, m*testMeses + b)
plt.show()
