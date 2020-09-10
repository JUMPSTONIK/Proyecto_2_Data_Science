install.packages("dummies")

library(dummies)

library(cluster)
library(e1071)
library(mclust)
library(fpc)
library(NbClust)
library(factoextra)

boneage <- read.csv("boneage.csv")


set.seed(123)

boneageDummies <- cbind(boneage, dummy(boneage$male, verbose = T))


km <- kmeans(boneageDummies %in% c(2,4,5), 1)

g1 <- km$cluster

prop.table(table(g1)) * 100

nrow(g1)

summary(g1)

#Jerarquico
hc <- hclust(dist(boneageDummies %in% c(2,4,5)))

plot(hc)
rect.hclust(hc, k=2)

groups <- cutree(hc, k=3)

g1HC <- boneageDummies[groups==1,]
g2HC <- boneageDummies[groups==2,]
g3HC <- boneageDummies[groups==3,]

plot()

#otro cluster
boneageDummies <- cbind(boneage, dummy(boneage$male, verbose = T))


mc <- Mclust(boneageDummies %in% c(2,4,5), 3)