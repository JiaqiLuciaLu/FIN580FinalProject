# rm(list = ls())
library("lattice")

# factor_path = '../data/factor/'
# setwd("~/YourWorkingDirectory")

combinetrees = function(feats_list, feat1, tree_depth, factor_path, tree_sort_path_base){
  # Combine portfolios from different trees and dedup the higher level trees
  print(feat1)
  feats = c('LME', feats_list[feat1])
  n_feats=length(feats)
  
  tree_sort_path = paste(tree_sort_path_base,paste(feats,collapse = '_'),'/',sep='')
  
  # result for tree sort
  feat_list_id_k = expand.grid(rep(list(1:n_feats),tree_depth))
  
  for (i in 1:n_feats){
    k=1
    file_name_id = paste(paste(feat_list_id_k[k,],collapse = ''),feats[i],'_min.csv',sep='')
    file_name = paste(tree_sort_path, file_name_id, sep='')
    port_ret0 = read.table(file_name, header=T, sep=',')
    colnames(port_ret0) = paste(paste(feat_list_id_k[k,],collapse = ''),substring(colnames(port_ret0),2),sep=".")
    port_ret = port_ret0
    
    for (k in 2:n_feats^tree_depth){
      file_name_id = paste(paste(feat_list_id_k[k,],collapse = ''),feats[i],'_min.csv',sep='')
      file_name = paste(tree_sort_path, file_name_id, sep='')
      port_ret0 = read.table(file_name, header=T, sep=',')
      colnames(port_ret0) = paste(paste(feat_list_id_k[k,],collapse = ''),substring(colnames(port_ret0),2),sep=".")
      port_ret = cbind(port_ret,port_ret0)
    }
    port_transpose=t(as.matrix(port_ret))
    keep = !duplicated(port_transpose)
    port_ret = port_ret[,keep] 
    print(ncol(port_ret))
    write.table(port_ret, paste(tree_sort_path,'/level', '_all_',feats[i],'_min.csv', sep=''), sep=',',row.names=F)
    
    k=1
    file_name_id = paste(paste(feat_list_id_k[k,],collapse = ''),feats[i],'_max.csv',sep='')
    file_name = paste(tree_sort_path, file_name_id, sep='')
    port_ret0 = read.table(file_name, header=T, sep=',')
    colnames(port_ret0) = paste(paste(feat_list_id_k[k,],collapse = ''),substring(colnames(port_ret0),2),sep=".")
    port_ret = port_ret0
    
    for (k in 2:n_feats^tree_depth){
      file_name_id = paste(paste(feat_list_id_k[k,],collapse = ''),feats[i],'_max.csv',sep='')
      file_name = paste(tree_sort_path, file_name_id, sep='')
      port_ret0 = read.table(file_name, header=T, sep=',')
      colnames(port_ret0) = paste(paste(feat_list_id_k[k,],collapse = ''),substring(colnames(port_ret0),2),sep=".")
      port_ret = cbind(port_ret,port_ret0)
    }
    port_ret = port_ret[,keep] 
    print(ncol(port_ret))
    write.table(port_ret, paste(tree_sort_path,'/level', '_all_',feats[i],'_max.csv', sep=''), sep=',',row.names=F)
  }
}
