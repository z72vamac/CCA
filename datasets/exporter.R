require(ggplot2)
# install.packages("GGally")
require(GGally)
require(CCA)
library(dplyr)
library(tidyverse)

unscaled_winequalityred <- read.csv("raw_winequalityred.csv", sep=";")
write.csv(unscaled_winequalityred[, 1:11], file = 'unscaled_winequalityred.csv', row.names = FALSE)
scaled_winequalityred = unscaled_winequalityred %>% mutate_all(~(scale(.) %>% as.vector))
write.csv(scaled_winequalityred[, 1:11], file = 'scaled_winequalityred.csv', row.names = FALSE)

unscaled_winequalitywhite <- read.csv("raw_winequalitywhite.csv", sep=';')
write.csv(unscaled_winequalitywhite[, 1:11], file = 'unscaled_winequalitywhite.csv')
scaled_winequalitywhite = unscaled_winequalitywhite %>% mutate_all(~(scale(.) %>% as.vector))
write.csv(scaled_winequalitywhite[, 1:11], file = 'scaled_winequalitywhite.csv', row.names = FALSE)

unscaled_studentmat <- read.csv("raw_studentmat.csv", sep=";", stringsAsFactors=TRUE)

unscaled_studentmat[, 1] = as.numeric(unscaled_studentmat[, 1] == unscaled_studentmat[1, 1])
unscaled_studentmat[, 2] = as.numeric(unscaled_studentmat[, 2] == unscaled_studentmat[1, 2])
unscaled_studentmat[, 4] = as.numeric(unscaled_studentmat[, 4] == unscaled_studentmat[1, 4])
unscaled_studentmat[, 5] = as.numeric(unscaled_studentmat[, 5] == unscaled_studentmat[1, 5])
unscaled_studentmat[, 6] = as.numeric(unscaled_studentmat[, 6] == unscaled_studentmat[1, 6])
unscaled_studentmat[, 16] = as.numeric(unscaled_studentmat[, 16] == "yes")
unscaled_studentmat[, 17] = as.numeric(unscaled_studentmat[, 17] == "yes")
unscaled_studentmat[, 18] = as.numeric(unscaled_studentmat[, 18] == "yes")
unscaled_studentmat[, 19] = as.numeric(unscaled_studentmat[, 19] == "yes")
unscaled_studentmat[, 20] = as.numeric(unscaled_studentmat[, 20] == "yes")
unscaled_studentmat[, 21] = as.numeric(unscaled_studentmat[, 21] == "yes")
unscaled_studentmat[, 22] = as.numeric(unscaled_studentmat[, 22] == "yes")
unscaled_studentmat[, 23] = as.numeric(unscaled_studentmat[, 23] == "yes")

write.csv(unscaled_studentmat[, c(1:8, 13:30)], file="unscaled_studentmat.csv", row.names = FALSE)
scaled_studentmat = unscaled_studentmat[,c(1:8, 13:30)] %>% mutate_all(~(scale(.) %>% as.vector))
write.csv(scaled_studentmat, file = 'scaled_studentmat.csv', row.names = FALSE)


unscaled_studentpor <- read.csv("raw_studentpor.csv", sep=";", stringsAsFactors=TRUE)

unscaled_studentpor[, 1] = as.numeric(unscaled_studentpor[, 1] == unscaled_studentpor[1, 1])
unscaled_studentpor[, 2] = as.numeric(unscaled_studentpor[, 2] == unscaled_studentpor[1, 2])
unscaled_studentpor[, 4] = as.numeric(unscaled_studentpor[, 4] == unscaled_studentpor[1, 4])
unscaled_studentpor[, 5] = as.numeric(unscaled_studentpor[, 5] == unscaled_studentpor[1, 5])
unscaled_studentpor[, 6] = as.numeric(unscaled_studentpor[, 6] == unscaled_studentpor[1, 6])
unscaled_studentpor[, 16] = as.numeric(unscaled_studentpor[, 16] == "yes")
unscaled_studentpor[, 17] = as.numeric(unscaled_studentpor[, 17] == "yes")
unscaled_studentpor[, 18] = as.numeric(unscaled_studentpor[, 18] == "yes")
unscaled_studentpor[, 19] = as.numeric(unscaled_studentpor[, 19] == "yes")
unscaled_studentpor[, 20] = as.numeric(unscaled_studentpor[, 20] == "yes")
unscaled_studentpor[, 21] = as.numeric(unscaled_studentpor[, 21] == "yes")
unscaled_studentpor[, 22] = as.numeric(unscaled_studentpor[, 22] == "yes")
unscaled_studentpor[, 23] = as.numeric(unscaled_studentpor[, 23] == "yes")

write.csv(unscaled_studentpor[, c(1:8, 13:30)], file="unscaled_studentpor.csv", row.names = FALSE)
scaled_studentpor = unscaled_studentpor[,c(1:8, 13:30)] %>% mutate_all(~(scale(.) %>% as.vector))
write.csv(scaled_studentpor, file = 'scaled_studentpor.csv', row.names = FALSE)

unscaled_music <- read.csv("raw_music.txt", header=FALSE)
write.csv(unscaled_music[, 1:68], file="unscaled_music.csv", row.names = FALSE)
scaled_music = unscaled_music %>% mutate_all(~(scale(.) %>% as.vector))
write.csv(scaled_music[, 1:68], file="scaled_music.csv", row.names = FALSE)

unscaled_yearprediction <- read.csv("raw_yearprediction.txt", header=FALSE)
write.csv(unscaled_yearprediction[, 2:91], file="unscaled_yearprediction.csv", row.names = FALSE)
scaled_yearprediction = unscaled_yearprediction %>% mutate_all(~(scale(.) %>% as.vector))
write.csv(scaled_yearprediction[, 2:91], file="scaled_yearprediction.csv", row.names = FALSE)

library(PMA)
library(tidyverse)

load(file='breastdata.rda')

set.seed(22) 
attach(breastdata)

dna <- t(dna)
rna <- t(rna)

unscaled_brst10a = as.data.frame(cbind(dna[, 1:10], rna[, 1:10]))
write.csv(unscaled_brst10a, file='unscaled_brst10a.csv', row.names = FALSE)
scaled_brst10a = unscaled_brst10a %>% mutate_all(~(scale(.) %>% as.vector))
write.csv(scaled_brst10a, file='scaled_brst10a.csv', row.names = FALSE)

unscaled_brst10a = as.data.frame(cbind(dna[, 1:10], rna[, 1:10]))
write.csv(unscaled_brst10a, file='unscaled_brst10a.csv', row.names = FALSE)
scaled_brst10a = unscaled_brst10a %>% mutate_all(~(scale(.) %>% as.vector))
write.csv(scaled_brst10a, file='scaled_brst10a.csv', row.names = FALSE)


write.csv(data.frame(dna[, 1:10]) %>% mutate_all(~(scale(.) %>% as.vector)), 'brst10a_dna_scaled.csv')
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

