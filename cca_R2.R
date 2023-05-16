library(data.table)
# library(tidyverse)
library(PMA)

set.seed(22) 

matriz_resultados = matrix(,nrow = 40, ncol = 8)

ks = c(2, 2, 3, 3, 2, 2, 3, 3, 3, 3, 5, 5, 3, 3, 5, 5)
(matriz_ks = t(matrix(ks, nrow = 2)))

n_solution = 1

n_row_start = 1
for (mode in 6:9)
{
  
  if (mode == 0 || mode == 5){n_row_finish = 3} else {n_row_finish = 2}
  if (mode == 0) {first_col = 1} else{first_col = 2}
  for (n_row in n_row_start:(n_row_start+n_row_finish-1))
    # for (n_row in n_row_start:n_row_start)
    {
      for (iter in 0:4)
        # for (iter in 0:4)
      {
        dataset1_train = read.csv(paste("~/Cosas de carlos/CCA/old_datasets/r_datasets/training_dataset1_mode_",mode,"_iteration_",iter,".csv", sep=""), header=TRUE)
        dims = dim(dataset1_train)
        dataset1_train = dataset1_train[ ,first_col:dims[2]]
        dataset1_test = read.csv(paste("~/Cosas de carlos/CCA/old_datasets/r_datasets/test_dataset1_mode_",mode,"_iteration_",iter,".csv", sep=""), header=TRUE)[,first_col:dims[2]]
        
        dataset2_train = read.csv(paste("~/Cosas de carlos/CCA/old_datasets/r_datasets/training_dataset2_mode_",mode,"_iteration_",iter,".csv", sep=""), header=TRUE)
        dims = dim(dataset2_train)
        dataset2_train = dataset2_train[ ,first_col:dims[2]]
        dataset2_test = read.csv(paste("~/Cosas de carlos/CCA/old_datasets/r_datasets/test_dataset2_mode_",mode,"_iteration_",iter,".csv", sep=""), header=TRUE)[,first_col:dims[2]]
        
        stop = FALSE
        
        penx = 0
        for (peny in seq(0, 1, 0.002))
        {
          # print(c(penx, peny))
          
          out <- CCA(x=dataset1_train,z=dataset2_train, typex="standard", typez="standard", penaltyx=penx, penaltyz=peny, trace = FALSE)
          # print(sum(out$v != 0))
          if (sum(out$v != 0) > matriz_ks[n_row, 1])
          {
            print("No hay valores deseados")
            break
          }
          if (sum(out$v != 0) == matriz_ks[n_row, 1])
          {
            for (penx in seq(0, 1, 0.002))
            {
              # print(c(penx, peny))
              start.time <- Sys.time()
              
              out <- CCA(x=dataset1_train,z=dataset2_train, typex="standard", typez="standard", penaltyx=penx, penaltyz=peny, trace = FALSE)
              end.time <- Sys.time()
              time.taken <- end.time - start.time
              
              if (sum(out$u != 0) == matriz_ks[n_row, 1])
              {
                print(n_row)
                stop = TRUE
                cor = out$cors
                print(cor)
                matriz_resultados[n_solution, 1] = n_solution
                matriz_resultados[n_solution, 2] = mode
                matriz_resultados[n_solution, 3] = 'lasso'
                matriz_resultados[n_solution, 4] = matriz_ks[n_row, 1]
                matriz_resultados[n_solution, 5] = matriz_ks[n_row, 1]
                matriz_resultados[n_solution, 6] = cor
                matriz_resultados[n_solution, 8] = time.taken
                break
              }
            }
          }
          if (stop){break}
        }
        
        matriz_test1 = as.matrix(dataset1_test)
        prod11 = t(matriz_test1) %*% matriz_test1
        norm1 = sqrt(t(out$u) %*% prod11 %*% out$u)
        
        matriz_test2 = as.matrix(dataset2_test)
        prod22 = t(matriz_test2) %*% matriz_test2
        norm2 = sqrt(t(out$v) %*% prod22 %*% out$v)
        
        prod12 = t(matriz_test1) %*% matriz_test2
        objval_test = (t(out$u) %*% prod12 %*% out$v)/(norm1*norm2)
        
        matriz_resultados[n_solution, 7] = objval_test
        n_solution = n_solution + 1
        
      }
      
      
    }
  n_row_start = n_row_start + n_row_finish
}

library(MASS)

write.matrix(matriz_resultados,file="matriz_resultados2.csv", sep=',')

