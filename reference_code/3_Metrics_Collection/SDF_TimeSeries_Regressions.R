## Source file to compute the statistics of the portfolios

### The basic input "port_ret" is always a T*N matrix, N is the number of portfolios and
### T is the length of time span. 
# rm(list = ls())
# setwd("~/YourWorkingDirectory")

FF_regression = function(port_ret, factor_path, option){
  ### Fama French regression
  ### First to choose the factor
  factor_file = paste(factor_path,'tradable_factors.csv',sep='')
  factor_mat = read.table(factor_file, header=T, sep=',')[361:636,]
  
  if(class(option)=='numeric'){
    factor = factor_mat[,option]
    X = as.matrix(cbind(rep(1,nrow(factor)),factor))
  }else if(option=='FF3'){
    factor = factor_mat[,2:4]
    X = as.matrix(cbind(rep(1,length(factor)),factor))
  }else if(option=='FF5'){
    factor = factor_mat[,c(2,3,4,6,7)]
    X = as.matrix(cbind(rep(1,nrow(factor)),factor))
  }else if(option=='FF11'){
    factor = factor_mat[,2:12]
    X = as.matrix(cbind(rep(1,nrow(factor)),factor))
  }
  
  model = lm(port_ret~X-1)
  oos = port_ret - X[,-1] %*% coef(summary(model))[-1,1]
  
  res = c(coef(summary(model))[1,1],coef(summary(model))[1,2], coef(summary(model))[1,3], coef(summary(model))[1,4])
  
  return(res)
}


compute_Statistics = function(port_ret, factor_path, option){
  ### Compute:
  ### Mean Absolute Alpha
  ### Maximum Sharpe
  ### GRS
  
  alpha = rep(0,4)
  se = rep(0,4)
  tStat = rep(0,4)
  pval = rep(0,4)
  
  ### FF3 regression
  res_FF = FF_regression(port_ret, factor_path, 'FF3')
  alpha[1] = res_FF[1]
  se[1] = res_FF[2]
  tStat[1] = res_FF[3]
  pval[1] = res_FF[4]
  ### FF5 regression
  res_FF = FF_regression(port_ret, factor_path, 'FF5')
  alpha[2] = res_FF[1]
  se[2] = res_FF[2]
  tStat[2] = res_FF[3]
  pval[2] = res_FF[4]
  ### XSF regression
  res_FF = FF_regression(port_ret, factor_path, option)
  alpha[3] = res_FF[1]
  se[3] = res_FF[2]
  tStat[3] = res_FF[3]
  pval[3] = res_FF[4]
  ### FF11 regression
  res_FF = FF_regression(port_ret, factor_path, 'FF11')
  alpha[4] = res_FF[1]
  se[4] = res_FF[2]
  tStat[4] = res_FF[3]
  pval[4] = res_FF[4]
  
  return(c(alpha, se, tStat, pval))
}


#################
### Main code ###
#################


results = matrix(nrow = 36, ncol = 16)

SDF_regression=function(feats_list, feat1, feat2, factor_path, port_path, port_name,weight_name){
  factors = c('Date','market',feats_list)
  T0 = 361
  T1 = 636
  
  feats_chosen = c('LME',feats_list[feat1],feats_list[feat2])
  print(feats_chosen)
  option = as.numeric(match(c('market',feats_chosen), factors))
  sub_dir = paste(feats_chosen,collapse = '_')
  
  port_ret = read.table(paste(port_path, sub_dir,port_name, sep=''), header=T, sep=',')[T0:T1, ]
  w = read.table(paste(port_path, sub_dir,weight_name,sep=''), header=T, sep=',')[,1]
  sdf = data.matrix(port_ret) %*% as.numeric(w)
  sdf = sdf/mean(sdf)
  
  result = compute_Statistics(sdf, factor_path, option)
  result = t(as.matrix(result))
  colnames(result) = c("FF3 Alpha", "FF5 Alpha", "XSF Alpha", "FF11 Alpha",
                       "FF3 SE", "FF5 SE", "XSF SE", "FF11 SE",
                       "FF3 T-Stat", "FF5 T-Stat", "XSF T-Stat", "FF11 T-Stat",
                       "FF3 P-val", "FF5 P-val", "XSF P-val", "FF11 P-val")
  dir.create(paste(port_path,'/SDFTests/',sep=''), showWarnings = FALSE)
  dir.create(paste(port_path,'/SDFTests/',sub_dir,sep=''), showWarnings = FALSE)
  write.table(result, paste(port_path,'/SDFTests/',sub_dir,'/TimeSeriesAlpha.csv', sep=''), sep=',',row.names=F) 
  return(result)
}


