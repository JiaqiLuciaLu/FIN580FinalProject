#rm(list = ls())
#setwd("~/YourWorkingDirectory")

# Parameters to change
# tree_depth = 4
# feats_list = c('LME','BEME','r12_2','OP','Investment','ST_Rev','LT_Rev','AC','IdioVol',"LTurnover")
# feat1 = 4
# feat2 = 5
# Generate value-weighted returns and min/max of characteristics of tree portfolios for one tree
library(dplyr)

create_tree_portfolio = function(y_min,y_max,tree_depth,feats_list,feat1,feat2,input_path,output_path,runparallel,pralleln){
  print(feat1)
  print(feat2)
  
  if (tree_depth == 3){
    cnames=c(1,11,12,111,112,121,122,1111,1112,1121,1122,1211,1212,1221,1222)
  }else if (tree_depth == 4){
    cnames=c(1,11,12,111,112,121,122,1111,1112,1121,1122,1211,1212,1221,1222,11111,11112,11121,11122,11211,11212,11221,11222,12111,12112,12121,12122,12211,12212,12221,12222)
  }else if (tree_depth == 5){
    cnames=c(1, 11, 12, 111, 112, 121, 122, 1111, 1112, 1121, 1122, 1211, 1212, 1221, 1222, 11111, 11112, 11121, 11122, 11211, 11212, 11221, 11222, 12111, 12112, 12121, 12122, 12211, 12212, 12221, 12222, 111111, 111112, 111121, 111122, 111211, 111212, 111221, 111222, 112111, 112112, 112121, 112122, 112211, 112212, 112221, 112222, 121111, 121112, 121121, 121122, 121211, 121212, 121221, 121222, 122111, 122112, 122121, 122122, 122211, 122212, 122221, 122222) 
  }
  
  feats = c('LME', feats_list[feat1])
  
  n_feats = length(feats)
  # main_dir = '../data/tree_portfolio_quantile'
  main_dir = output_path
  sub_dir = paste(feats,collapse = '_')
  dir.create(file.path(main_dir, sub_dir), showWarnings = FALSE)
  # data_path = paste('../data/data_chunk_files_quantile/',paste(feats,collapse = '_'),'/', sep='')
  data_path = paste(input_path,paste(c('LME', feats_list[feat1],feats_list[feat2]),collapse = '_'),'/', sep='')
  
  q_num = 2
  
  feat_list_base = feats
  feat_list_id_k = expand.grid(rep(list(1:n_feats),tree_depth))

  if (runparallel){
    library(doSNOW)
    library(foreach)
    
    cl<-makeCluster(pralleln)
    registerDoSNOW(cl)
    foreach(k=c(1:n_feats^tree_depth),.packages='dplyr') %dopar% {
      source('1_Portfolio_Creation/Tree_Portfolio_Creation/tree_portfolio_helper.R', local = TRUE)
      file_id = paste(feat_list_id_k[k,],collapse = '')
      feat_list = feat_list_base[as.numeric(feat_list_id_k[k,])]
      ret = tree_portfolio(data_path, feat_list, tree_depth, q_num, y_min, y_max, 'y',feats)
      ret_table = ret[[1]]
      
      colnames(ret_table)=cnames

      write.table(ret_table, paste(file.path(main_dir, sub_dir),'/', file_id,'ret.csv', sep=''), sep=',',row.names=F) 
      
      feat_min_table = list()
      feat_max_table = list()
      for (f in 1:n_feats){
        feat_min_table[[f]] = ret[[2*f]]
        colnames(feat_min_table[[f]])=cnames
        write.table(feat_min_table[[f]], paste(file.path(main_dir, sub_dir),'/', file_id,feats[f],'_min.csv', sep=''), sep=',',row.names=F) 
        
        feat_max_table[[f]] = ret[[2*f+1]]
        colnames(feat_max_table[[f]])=cnames
        write.table(feat_max_table[[f]], paste(file.path(main_dir, sub_dir),'/', file_id,feats[f],'_max.csv', sep=''), sep=',',row.names=F) 
      }
    }
    stopCluster(cl)
  }else{
    source('1_Portfolio_Creation/Tree_Portfolio_Creation/tree_portfolio_helper.R', local = TRUE)
    for (k in c(1:n_feats^tree_depth)){
      print(k)
      file_id = paste(feat_list_id_k[k,],collapse = '')
      feat_list = feat_list_base[as.numeric(feat_list_id_k[k,])]
      ret = tree_portfolio(data_path, feat_list, tree_depth, q_num, y_min, y_max, 'y',feats)
      ret_table = ret[[1]]
      
      colnames(ret_table)=cnames
      
      write.table(ret_table, paste(file.path(main_dir, sub_dir),'/', file_id,'ret.csv', sep=''), sep=',',row.names=F) 
      
      feat_min_table = list()
      feat_max_table = list()
      for (f in 1:n_feats){
        feat_min_table[[f]] = ret[[2*f]]
        colnames(feat_min_table[[f]])=cnames
        write.table(feat_min_table[[f]], paste(file.path(main_dir, sub_dir),'/', file_id,feats[f],'_min.csv', sep=''), sep=',',row.names=F) 
        
        feat_max_table[[f]] = ret[[2*f+1]]
        colnames(feat_max_table[[f]])=cnames
        write.table(feat_max_table[[f]], paste(file.path(main_dir, sub_dir),'/', file_id,feats[f],'_max.csv', sep=''), sep=',',row.names=F) 
      }
    }
  }
}



