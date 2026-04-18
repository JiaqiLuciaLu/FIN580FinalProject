source('3_Metrics_Collection/Pick_Best_Lambda.R')

pickSRN=function(feats_list,feat1,feat2,grid_search_path,mink,maxk,lambda0,lambda2,port_path,port_file_name){
  feats = c('LME', feats_list[feat1], feats_list[feat2])
  subdir = paste(feats, collapse = '_')
  
  srN = pickBestLambda(feats_list,feat1,feat2, grid_search_path,mink,lambda0,lambda2, port_path,port_file_name,FALSE)
  for (k in (mink+1):maxk){
    print(k)
    srN = cbind(srN, pickBestLambda(feats_list,feat1,feat2, grid_search_path,k,lambda0,lambda2, port_path,port_file_name,FALSE))
  }
  write.table(srN, paste(grid_search_path, subdir, '/SR_N.csv', sep=''), sep=',',row.names=F)
}