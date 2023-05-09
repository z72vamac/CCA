require(ggplot2)
# install.packages("GGally")
require(GGally)
require(CCA)
library(dplyr)
library(tidyverse)




mm <- read.csv("mmreg.csv")
colnames(mm) <- c("Control", "Concept", "Motivation", "Read", "Write", "Math", 
                  "Science", "Sex")
summary(mm)

mm <- mm %>% mutate_all(~(scale(.) %>% as.vector))

write.csv(mm, file="mmreg_scaled.csv", row.names = FALSE)

psych <- mm[, 1:3]
acad <- mm[, 4:8]

# d = cc(X1, X2)

cc1 <- cc(psych, acad)

# display the canonical correlations
cc1$cor
cc1$xcoef
cc1$ycoef


kk = cancor(psych, acad, xcenter = F, ycenter = F)
kk$cor


# if (!requireNamespace("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# 
# BiocManager::install("impute")
# 
# install.packages("PMA")


library(PMA)
cc2 = CCA(psych, acad, K = 1, penaltyx = 1, penaltyz = 0.5)
names(cc2)

summary(cc2)
cc2$u
cc2$v

cc2

library(calibrate)

cc3 = canocor(psych, acad)

cc3


# winequality.white <- read.csv("C:/Users/Justo Puerto/UNIVERSIDAD DE SEVILLA/Grupo de Trabajo de Justo - General/CCA/winequality-white.csv", sep=";")
# winequality.red <- read.csv("C:/Users/Justo Puerto/UNIVERSIDAD DE SEVILLA/Grupo de Trabajo de Justo - General/CCA/YearPredictionMSD.txt", header=FALSE)[,2:91]
# 
# write.csv(winequality.red, file="yearprediction.csv", row.names = FALSE)
# 
# winequality.red = winequality.red %>% mutate_all(~(scale(.) %>% as.vector))
# 
# write.csv(winequality.red, file="yearprediction_scaled.csv", row.names = FALSE)

student.mat <- read.csv("C:/Users/Justo Puerto/UNIVERSIDAD DE SEVILLA/Grupo de Trabajo de Justo - General/CCA/student-por.csv", sep=";", stringsAsFactors=TRUE)

write.csv(student.mat, file="student.mat.csv", row.names = FALSE)


student.mat[, 1] = as.numeric(student.mat[, 1] == student.mat[1, 1])
student.mat[, 2] = as.numeric(student.mat[, 2] == student.mat[1, 2])
student.mat[, 4] = as.numeric(student.mat[, 4] == student.mat[1, 4])
student.mat[, 5] = as.numeric(student.mat[, 5] == student.mat[1, 5])
student.mat[, 6] = as.numeric(student.mat[, 6] == student.mat[1, 6])
student.mat[, 12] = as.numeric(student.mat[, 12] == "yes")
student.mat[, 13] = as.numeric(student.mat[, 13] == "yes")
student.mat[, 14] = as.numeric(student.mat[, 14] == "yes")
student.mat[, 15] = as.numeric(student.mat[, 15] == "yes")
student.mat[, 16] = as.numeric(student.mat[, 16] == "yes")
student.mat[, 17] = as.numeric(student.mat[, 17] == "yes")
student.mat[, 18] = as.numeric(student.mat[, 18] == "yes")
student.mat[, 19] = as.numeric(student.mat[, 19] == "yes")


student.mat

student.mat = student.mat %>% mutate_all(~(scale(.) %>% as.vector))

student.mat
write.csv(student.mat, file="studentpor_scaled.csv", row.names = FALSE)

# music = read.csv("C:/Users/Justo Puerto/UNIVERSIDAD DE SEVILLA/Grupo de Trabajo de Justo - General.txt", header=FALSE)[,1:68]
# 
# write.csv(music, file="music.csv", row.names = FALSE)
# 
# music = music %>% mutate_all(~(scale(.) %>% as.vector))
# 
# write.csv(music, file="music_scaled.csv", row.names = FALSE)

X = music[,1:34]
Y = music[,35:68]


cc_new = cc(X, Y)
cc_new$cor


library(data.table)
library(tidyverse)

breastdata_rna <- read.csv("C:/Users/Justo Puerto/UNIVERSIDAD DE SEVILLA/Grupo de Trabajo de Justo - General/CCA/datasets/breastcancer/breastdata_rna.txt", header=FALSE)
breastdata_dna <- read.csv("C:/Users/Justo Puerto/UNIVERSIDAD DE SEVILLA/Grupo de Trabajo de Justo - General/CCA/datasets/breastcancer/breastdata_dna.txt", header=FALSE)



breastdata_rna = data.frame(transpose(breastdata_rna))
breastdata_rna = breastdata_rna %>% mutate_all(~(scale(.) %>% as.vector))

breastdana_dna = transpose(breastdata_dna)
breastdata_dna = breastdata_dna %>% mutate_all(~(scale(.) %>% as.vector))



library(PMA)
library(tidyverse)

load(file='C:\\Users\\Justo Puerto\\UNIVERSIDAD DE SEVILLA\\Grupo de Trabajo de Justo - General\\CCA\\breastdata.rda')

set.seed(22) 
# data(breastdata) 
attach(breastdata)

dna <- t(dna)
rna <- t(rna)


write.csv(data.frame(dna[, 1:10]) %>% mutate_all(~(scale(.) %>% as.vector)), 'datasets/brst10a_dna_scaled.csv')
write.csv(data.frame(rna[, 1:10]) %>% mutate_all(~(scale(.) %>% as.vector)), 'datasets/brst10a_rna_scaled.csv')

write.csv(data.frame(dna[, 11:20]) %>% mutate_all(~(scale(.) %>% as.vector)), 'datasets/brst10b_dna_scaled.csv')
write.csv(data.frame(rna[, 11:20]) %>% mutate_all(~(scale(.) %>% as.vector)), 'datasets/brst10b_rna_scaled.csv')

write.csv(data.frame(dna[, 1:20]) %>% mutate_all(~(scale(.) %>% as.vector)), 'datasets/brst20a_dna_scaled.csv')
write.csv(data.frame(rna[, 1:20]) %>% mutate_all(~(scale(.) %>% as.vector)), 'datasets/brst20a_rna_scaled.csv')

write.csv(data.frame(dna[, 21:40]) %>% mutate_all(~(scale(.) %>% as.vector)), 'datasets/brst20b_dna_scaled.csv')
write.csv(data.frame(rna[, 21:40]) %>% mutate_all(~(scale(.) %>% as.vector)), 'datasets/brst20b_rna_scaled.csv')

# write.csv(data.frame(dna[, 1:10]), 'datasets/brst10a_dna_scaled.csv')
# write.csv(data.frame(rna[, 1:10]), 'datasets/brst10a_rna_scaled.csv')
# 
# write.csv(data.frame(dna[, 11:20]), 'datasets/brst10b_dna_scaled.csv')
# write.csv(data.frame(rna[, 11:20]), 'datasets/brst10b_rna_scaled.csv')
# 
# write.csv(data.frame(dna[, 1:20]), 'datasets/brst20a_dna_scaled.csv')
# write.csv(data.frame(rna[, 1:20]), 'datasets/brst20a_rna_scaled.csv')
# 
# write.csv(data.frame(dna[, 21:40]), 'datasets/brst20b_dna_scaled.csv')
# write.csv(data.frame(rna[, 21:40]), 'datasets/brst20b_rna_scaled.csv')



for (i in 1:23)
{
  dna_i <- data.frame(dna[, chrom==i])
  dna_i = dna_i %>% mutate_all(~(scale(.) %>% as.vector))
  write.csv(dna_i, paste('datasets/dna_scaled',i, '.csv', sep=""))
}



for (i in 1:23)
{
  rna_i <- data.frame(rna[, chrom==i])
  rna_i = rna_i %>% mutate_all(~(scale(.) %>% as.vector))
  write.csv(rna_i, paste('datasets/rna_scaled',i, '.csv', sep=""))
}

# rna <- data.frame(rna)
# rna = rna %>% mutate_all(~(scale(.) %>% as.vector))
# write.csv(rna, 'datasets/rna_scaled.csv')

# genechr = data.frame(genechr)[, 1]
# write.csv(genechr, "datasets/genechr.csv",row.names=F)

CCA.permute 
perm.out <- CCA.permute(x=rna,z=dna[,chrom==1],typex="standard", typez="ordered",nperms=5,penaltyxs=seq(.02,.7,len=10)) 
## We run CCA using all gene exp. data, but CGH data on chrom 1 only. 
print(perm.out)
plot(perm.out) 
out <- CCA(x=rna,z=dna[,chrom==1], typex="standard", typez="ordered", penaltyx=perm.out$bestpenaltyx, v=perm.out$v.init, penaltyz=perm.out$bestpenaltyz, xnames=substr(genedesc,1,20), znames=paste("Pos", sep="", nuc[chrom==1])) 
# Save time by inputting lambda and v 
print(out) # could do print(out,verbose=TRUE) 
print(genechr[out$u!=0]) 
# Cool! The genes associated w/ gain or loss ## on chrom 1 are located on chrom 1!! 
par(mfrow=c(1,1)) 
PlotCGH(out$v, nuc=nuc[chrom==1], chrom=chrom[chrom==1], main="Regions of gain/loss on Chrom 1 assoc'd with gene expression") 



set.seed(22)
load(file='C:\\Users\\Justo Puerto\\UNIVERSIDAD DE SEVILLA\\Grupo de Trabajo de Justo - General\\CCA\\breastdata.rda')
attach(breastdata) 
dna <- t(dna) 
rna <- t(rna) 
perm.out <- CCA.permute(x=rna,z=dna[,chrom==2],typex="standard", typez="ordered",nperms=25,penaltyxs=seq(.02,.7,len=10)) ## We run CCA using all gene exp. data, but CGH data on chrom 1 only. 
print(perm.out) 
plot(perm.out) 
out <- CCA(x=rna,z=dna[,chrom==1], typex="standard", typez="ordered", penaltyx=perm.out$bestpenaltyx, v=perm.out$v.init, penaltyz=perm.out$bestpenaltyz, xnames=substr(genedesc,1,20), znames=paste("Pos", sep="", nuc[chrom==1]))

# Save time by inputting lambda and v 
print(out) # could do print(out,verbose=TRUE) 
print(genechr[out$u!=0]) # Cool! The genes associated w/ gain or loss ## on chrom 1 are located on chrom 1!!


genechr[out$u != 0]

for (i in 1:23){
  print(sum(genechr == i, na.rm = TRUE))
}

