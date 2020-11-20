# Proyecto_2_Data_Science

Como Ejecutar aplicación de Dash:

Tener instaladas lis librerías:
-numpy
-pandas
-keras
-tensorflow
-plotly
-dash
-matplotlib
-seaborn
-sklearn
-hspy

Se pueden instalar por separado pero es preferible por medio del comando
pip install -r requirements.txt
en la carpeta raíz del repositorio

Tener en el mismo directorio los archivos:
AplicacionDash.py
boneage-training-dataset.csv
ImagenRegresionSecuencial15.png
modeloXceptionScatter.jpeg
modelo.h5 (Se puede obtener en el siguiente link: https://drive.google.com/file/d/19LR7z8tEi_UR1XZtRkPC6yuDm0vfZmf3/view?usp=sharing)
seqModelo_weights_lineal_15.h5 (Se puede obtener en el siguiente link: https://drive.google.com/drive/folders/1yhzsZa7QtmRpAN82CO_mh_vYlBvg7Ftl?usp=sharing)

FInalmente ejecutar el programa AplicacionDash.py
puede ser con el comando python AplicacionDash.py o ejecutando el .py desde la carpeta.

El ejecutar puede o no abrir el navegador en la url "http://127.0.0.1:8050/"
Si no se abre, copiar y pegar "http://127.0.0.1:8050/" en el navegador

Como utilizar.

Para utilizar el modelo secuencial se puede arrastrar una imagen desde el explorador de archivos o se puede hacer click en el recuadro y seleccionar una imagen. Esto desplegará la imagen debajo del recuadro y dirá la predicción de edad. Si se desea saber la eefectividad del modelo y visualizar la grafica de regresión de la misma se debe hacer click en el textbox en la esquina inferior derecha del recuadro celeste. Cuando ya no se desee visualizar deseleccionar el checkbox

Para utilizar el modelo Xception se debe tener en la misma carpeta donde se ejecuta el programa la imagen a evaliar, luego en el textbox dentro del recuadro amarillo se debe escribir el nombre del archivo de imagen, inclutendo la extension. Luego se presiona el boton "Submit". Esto desplegará la imagen debajo del recuadro y dirá la predicción de edad. Si se desea saber la efectividad del modelo y visualizar la grafica de regresión de la misma se debe hacer click en el textbox en la esquina inferior derecha del recuadro amarillo. Cuando ya no se desee visualizar deseleccionar el checkbox