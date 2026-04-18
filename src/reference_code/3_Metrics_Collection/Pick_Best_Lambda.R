
pickBestLambda=function(feats_list, feat1, feat2, ap_prune_result_path,portN,lambda0,lambda2, portfolio_path, port_name, fullCV=FALSE, writetable=TRUE){
  print(feat1)
  print(feat2)
  feats = c('LME', feats_list[feat1], feats_list[feat2])
  
  subdir = paste(feats, collapse = '_')
  
  train_SR = matrix(0, nrow = length(lambda0), ncol = length(lambda2))
  valid_SR = matrix(0, nrow = length(lambda0), ncol = length(lambda2))
  test_SR = matrix(0, nrow = length(lambda0), ncol = length(lambda2))
  
  for (i in 1:length(lambda0)){
    for (j in 1:length(lambda2)){
      full_data = read.csv(paste(ap_prune_result_path, subdir, "/results_full_l0_",i,"_l2_",j,".csv", sep=""), stringsAsFactors=FALSE)
      cv_data = read.csv(paste(ap_prune_result_path, subdir, "/results_cv_3_l0_",i,"_l2_",j,".csv", sep=""), stringsAsFactors=FALSE)
      
      train_SR[i,j] = full_data[full_data$portsN == portN,][1,1]
      valid_SR[i,j] = cv_data[cv_data$portsN == portN,][1,2]
      test_SR[i,j] = full_data[full_data$portsN == portN,][1,2]
      
      if (fullCV){
        cv_data = read.csv(paste(ap_prune_result_path, subdir, "/results_cv_1_l0_",i,"_l2_",j,".csv", sep=""), stringsAsFactors=FALSE)
        valid_SR[i,j] = valid_SR[i,j]+cv_data[cv_data$portsN == portN,][1,2]
        cv_data = read.csv(paste(ap_prune_result_path, subdir, "/results_cv_2_l0_",i,"_l2_",j,".csv", sep=""), stringsAsFactors=FALSE)
        valid_SR[i,j] = valid_SR[i,j]+cv_data[cv_data$portsN == portN,][1,2]
        valid_SR[i,j] = valid_SR[i,j]/3.0
      }
    }
  }
  
  index = which (valid_SR == max(valid_SR), arr.ind = TRUE)
  
  full_data = read.csv(paste(ap_prune_result_path, subdir, "/results_full_l0_",index[1],"_l2_",index[2],".csv", sep=""), stringsAsFactors=FALSE)
  
  weights = full_data[full_data$portsN == portN,][1,-(1:3)]
  weights_out = weights[weights!=0]
  ports_no = which(weights!=0)
  ports = read.csv(paste(portfolio_path, subdir, port_name, sep=""), stringsAsFactors=FALSE)
  selected_ports = ports[,ports_no]
  
  if (writetable){
    write.table(train_SR, paste(ap_prune_result_path, subdir, '/train_SR_',portN,'.csv', sep=''), sep=',',row.names=F)
    write.table(valid_SR, paste(ap_prune_result_path, subdir, '/valid_SR_',portN,'.csv', sep=''), sep=',',row.names=F)
    write.table(test_SR, paste(ap_prune_result_path, subdir, '/test_SR_',portN,'.csv', sep=''), sep=',',row.names=F)
    write.table(selected_ports, paste(ap_prune_result_path, subdir, '/Selected_Ports_',portN,'.csv', sep=''), sep=',',row.names=F)
    write.table(weights_out, paste(ap_prune_result_path, subdir, '/Selected_Ports_Weights_',portN,'.csv', sep=''), sep=',',row.names=F)
  }

  return(c(train_SR[index[1],index[2]],valid_SR[index[1],index[2]],test_SR[index[1],index[2]]))
}