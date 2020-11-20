import numpy as np
import pandas as pd
import os
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, GlobalMaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from tkinter import filedialog
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import datetime
import math
import seaborn as sb
from sklearn.model_selection import train_test_split
from keras.applications.xception import preprocess_input
from keras.metrics import mean_absolute_error
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import Xception
import time
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_fscore_support
import csv

train_df = pd.read_csv("boneage-training-dataset.csv")
train = train_df.head(10000)
test = train_df.tail(2600)
quadratic = True
image_size = 128

# some kind of ratio if not quadratic approach
img_rows = image_size if quadratic else 144 
img_cols = image_size if quadratic else 114
load_15_age_model = Sequential()
load_15_age_model.add(Conv2D(16, kernel_size=(7,7), strides=(2,2), activation='relu', input_shape=(img_rows, img_cols, 3)))
load_15_age_model.add(Dropout(0.25))
# Convulotional Layers
load_15_age_model.add(Conv2D(32, kernel_size=(5,5), strides=(2,2), activation='relu'))
load_15_age_model.add(Dropout(0.25))
load_15_age_model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
#bone_age_model.add(Dropout(0.25))
load_15_age_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
load_15_age_model.add(Dropout(0.25))
# Flattening
load_15_age_model.add(Flatten())
# Dense Layer
load_15_age_model.add(Dense(256, activation='relu'))
#bone_age_model.add(Dropout(0.25))
load_15_age_model.add(Dense(1, activation="linear"))
load_15_age_model.load_weights('seqModelo_weights_lineal_15.h5')

imgSize = 256
media = 127.3207517246848
desviacionS = 41.182021399396326

ba = float(1)
male = True
path = './'

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

def Prediciendo(nombre):

    df = pd.DataFrame(columns=['id', 'boneage', 'male'])
    df.loc[len(df.index)] = [nombre] + [float(ba)] + [male]
    df['val_z'] = (df['boneage'] - media) / desviacionS
    df['gender'] = df['male'].apply(lambda x: 'male' if x else 'female')

    validationDataGenerator = ImageDataGenerator(preprocessing_function = preprocess_input)


    testX, testY = next(validationDataGenerator.flow_from_dataframe(dataframe = df, directory = path, x_col = 'id', y_col = 'val_z', target_size = (imgSize, imgSize), batch_size = 2523, class_mode = 'raw'))

    def mae(xP, yP):
        return mean_absolute_error((desviacionS * xP + media), (desviacionS * yP + media))

    start_time = time.time()

    prediccion = media + desviacionS * (modelo2.predict(testX, batch_size = 32, verbose = True))

    return prediccion[0][0]
import base64
app = dash.Dash(__name__)
app.layout = html.Div([

    html.H1("Boneage", style={'text-align': 'left'}),

    html.Div(id='Sequential',style=({'background-color': '#c7fff8'}), children=[
        html.H2("Secuencial"),
        html.Div(id='Sequential_testing',children=[
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or Click to ',
                    html.A('Select File')
                ]),
                style={
                    'width': '20%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=False
            ),
            html.Div(id='output-data-upload'),
        ]),
        html.Div(id='Sequential_efectiveness',children=[
            dcc.Checklist(id='seq_check',
                options=[
                    {'label': 'Visualizar info del modelo', 'value': '1'}
                ]
            ),
            html.Div(id='Seq_data')
        ])
    ]),
    html.Div(id='Xception',style=({'background-color': '#fff5ad'}), children=[
        html.H2("Xception"),
        html.Div(id='Xception_testing',children=[
            html.Div(dcc.Input(id='input-box', type='text')),
            html.Button('Submit', id='button'),
            html.Div(id='output-container-button',
                     children='Enter a value and press submit')
        ]),
        html.Div(id='Xception_efectiveness',children=[
            dcc.Checklist(id='Xception_check',
                options=[
                    {'label': 'Visualizar info del modelo', 'value': '1'}
                ]
            ),
            html.Div(id='Xception_data')
        ])
    ])
    
])




def getPlot():
    df=pd.DataFrame({
        'Boneage': test_list,
        'Prediction': res
    })
    fig = px.scatter(data_frame =df,x='Boneage',y='Prediction')
    return fig
def evaluate_image(data):
    from PIL import Image
    from io import BytesIO
    content_type, content_string = data.split(',')

    im = Image.open(BytesIO(base64.b64decode(content_string))).resize((128,128)).convert('RGB')
    input_arr = keras.preprocessing.image.img_to_array(im)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    input_arr=input_arr/256
    predictions = load_15_age_model.predict(input_arr,  verbose = True)[0][0]
    return predictions
@app.callback(Output('Seq_data', 'children'),
    [Input('seq_check', 'value')])
def show_seq(values):
    if (len(values)>0):
        encoded_image = base64.b64encode(open("./ImagenRegresionSecuencial15.png", 'rb').read())
        children=[
            html.P(["Error medio absoluto: 27.349"]),
            html.P(["Error cuadrado medio: 1154.585"]),
            html.Img(src='data:image/png;base64,'+str(encoded_image.decode()),style={'height':'30%', 'width':'30%'})
            ]
        return children
@app.callback(Output('Xception_data', 'children'),
    [Input('Xception_check', 'value')])
def show_Xcep(values):
    if (len(values)>0):
        encoded_image = base64.b64encode(open("./modeloXceptionScatter.jpeg", 'rb').read())
        children=[
            html.P(["Error medio absoluto: 8.534"]),
            html.P(["Error cuadrado medio: 127.967"]),
            html.Img(src='data:image/png;base64,'+str(encoded_image.decode()),style={'height':'30%', 'width':'30%'})
        ]
        return children
@app.callback(
    dash.dependencies.Output('output-container-button', 'children'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('input-box', 'value')])
def update_Xception(n_clicks, value):
    encoded_image = base64.b64encode(open("./"+value, 'rb').read())
    return [
        html.P(["Prediccion: ",(Prediciendo(value))]),
        html.Img(src='data:image/png;base64,'+str(encoded_image.decode()),style={'height':'10%', 'width':'10%'})
           ]    
@app.callback(Output('output-data-upload', 'children'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'),
    State('upload-data', 'last_modified')])
def update_Sequential(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        predicted=evaluate_image(list_of_contents)
        children = [
            html.P(["Predicted boneage: ",predicted]),
            html.Img(src=list_of_contents,style={'height':'10%', 'width':'10%'})
        ]
        del list_of_contents, list_of_names, list_of_dates
        return children
if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
load_15_age_model.compile(loss='mean_squared_error', optimizer='adam', metrics='accuracy')
