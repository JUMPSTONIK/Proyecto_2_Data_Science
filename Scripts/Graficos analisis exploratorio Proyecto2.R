library(ggplot2)
library(readr)
library(dplyr)
#leemos el dataset
data <- read_csv("boneage-training-dataset.csv")
#calculamos los porcentajes de la cantidad de hombres y no hombres para las graficas de pie y porcentajes
porcentajes <- as.numeric(round(((prop.table(table(data$male)))*100),2))
#creamos el factor entre los valores de male con las etiquetas para hacer mas facil el visualizar la data
sexo_factor = factor(data$male,labels = c("Female", "Male"))
#Etiquetas para el gráfico
etiquetas <- c("female", "Male")
etiquetas <- paste(etiquetas, porcentajes)
#calculamos la densidad para los graficos de densidad
densidad_bonage <- density(data$boneage)
#histograma simple de 0 para no male(female) y 1 para male
hist(as.numeric(data$male), main = "Grafica de barras en crudo de male",xlab = "Sexo", ylab = "cantidad")
#histograma de frecuencia de female y male
plot(sexo_factor, main = "Gráfico de barras 1 de male",
     xlab = "Género", ylab = "Frecuencia")
#grafico de pie para representar los porcentajes de female y male
pie(porcentajes, etiquetas,
    main = "Gráfico de porcentaje de males y females",
    sub = "Evaluación el porcentaje de niños y niñas en el dataset")
#histograma para las edades en meses de boneage
hist(data$boneage, main = "Histograma de frecuencias 1 de boneage",
     xlab = "Edad(meses)",
     ylab = "Frecuencia",
     col = "blue",
     border = "black",
     xlim = c(0, 250),
     ylim = c(0, 3000))

#grafico de densidad para boneage
plot(densidad_bonage, 
     main = "Histograma de densidad de boneage",
     xlab = "Edad (meses)",
     ylab = "Densidad")
#diagrama de caja y bigotes de boneage
boxplot(data$boneage, main = "Gráfico de cajas de Boneage",
        outline = TRUE)
#histograma 2 de female y male mas bonito
ggplot(data, aes(x = sexo_factor)) +
  geom_bar(width = 0.4,  fill=rgb(0.1,1,0.5,0.7)) +
  scale_x_discrete("Sexo") +     # configuración eje X (etiqueta del eje)
  scale_y_continuous("Frecuencia") +
  labs(title = "Gráfico de barra 2 de male",
       subtitle = "Frecuencia absoluta de la male en forma de male y female")
#histograma 3 de male de los porcentajes de female y male
ggplot(data, aes(x = sexo_factor)) +
  geom_bar(width = 0.4, fill=rgb(0.1,0.3,0.5,0.7), aes(y = (..count..)/sum(..count..))) +
  scale_x_discrete("Sexo") +     # configuración eje X (etiqueta del eje)
  scale_y_continuous("Porcentaje",labels=scales::percent) + #Configuración eje y
  labs(title = "Gráfico de barras 3 de male",
       subtitle = "Frecuencia relativa de la variable sexo")
#histograma de frecuencias de boneage mas detallado
ggplot(data, aes(x = as.numeric(data$boneage))) +
  geom_histogram(binwidth = 0.6) +
  scale_x_continuous("Edad (meses)") + 
  scale_y_continuous("Frecuencia") +
  labs(title = "Histograma de frecuencias 2 de boneage",
       subtitle = "Frecuencia absoluta de la variable boneage")
#grafico 2 de densidad de boneage mas bonito
ggplot(data, aes(x = data$boneage)) +
  geom_density() +
  scale_y_continuous("Densidad") +
  scale_x_continuous("Edad(meses)") +
  labs(title = "Histograma de densidad 2 de boneage",
       subtitle = "Forma de la distribución de la variable boneage")
