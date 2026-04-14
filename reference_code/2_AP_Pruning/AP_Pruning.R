# rm(list = ls())
# setwd(substring(dirname(rstudioapi::getSourceEditorContext()$path), 1, nchar(dirname(rstudioapi::getSourceEditorContext()$path)) - 6))
source('2_AP_Pruning/lasso_valid_par_full.R')

AP_Pruning=function(feats_list, feat1, feat2, input_path,input_file_name, output_path,n_train_valid,cvN,runFullCV,kmax,RunParallel, ParallelN,IsTree,lambda0,lambda2){
  chars = c('LME', feats_list[feat1], feats_list[feat2])
  subdir = paste(chars, collapse = '_')
  print(subdir)
  
  ports <- read.csv(paste(input_path, subdir, input_file_name, sep=""), stringsAsFactors=FALSE)
  if (IsTree){
    depths = nchar(colnames(ports)) - 7
    coln = colnames(ports)
    for (i in 1:length(coln)){
      coln[i] = paste(substring(coln[i],2,nchar(substring(coln[i], 7, 11))), substring(coln[i], 7, 11), sep='.')
    }
    colnames(ports) = coln
  }else{
    depths = rep(0, ncol(ports))
  }
  
  adj_w = 1/sqrt(2^depths)
  
  adj_ports = ports
  if (IsTree){
    for (i in 1:length(adj_w)){
      adj_ports[,i] = ports[,i] * adj_w[i]
    }
  }
  
  lasso_valid_full(adj_ports, lambda0, lambda2, output_path, subdir, adj_w, n_train_valid, cvN, runFullCV,kmax=kmax, RunParallel=RunParallel,ParallelN=ParallelN)
}
