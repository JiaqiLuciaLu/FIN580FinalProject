# setwd("~/YourWorkingDirectory")
library('dplyr')

triple_sort = function(data_path, feat1, feat2, feat3, y_min, y_max){
  ret_table = array(0,dim=c(64,(y_max-y_min+1)*12))
  y_time_stamp = 1
  for(y in c(y_min:y_max)){
    print(y)
    data_filenm = paste(data_path,'y',toString(y),'.csv',sep='')
    df_tmp = read.csv(data_filenm)
    ret_table[,c(((y_time_stamp-1)*12+1):(y_time_stamp*12))] = triple_sort_helper(df_tmp, feat1, feat2, feat3) 
    y_time_stamp = y_time_stamp+1
  }  
  return(ret_table)
}

triple_sort_helper = function(df_tmp, feat1, feat2, feat3){
  ret_tmp = array(0,dim=c(64,12))
  df_tmp['1'] = 0
  df_tmp['2'] = 0
  df_tmp['3'] = 0
  for(mon in c(1:12)){
    mask_m = (df_tmp['mm']==mon)
    df_m = df_tmp[mask_m,]
    df_m['1'] = ntile(df_m[feat1], 4)
    df_m['2'] = ntile(df_m[feat2], 4)
    df_m['3'] = ntile(df_m[feat3], 4)
    for(i in c(1:4)){
      for(j in c(1:4)){
        for(k in c(1:4)){
          mask_port = (df_m['1']==i) & (df_m['2']==j) & (df_m['3']==k)
          company_val = df_m['size'][mask_port,]
          ret_mon = df_m['ret'][mask_port,]
          ret_tmp[(i-1)*16+(j-1)*4 + k,mon] = ret_mon%*%company_val/sum(company_val)
        } 
      }
    }
  }
  return(ret_tmp)
}

remove_rf = function(port_ret, factor_path){
  file_nm = paste(factor_path,'rf_factor.csv',sep='')
  r_f = read.table(file_nm, header = F, sep=',')
  for(i in c(1:ncol(port_ret))){
    port_ret[,i] = port_ret[,i]-(as.numeric(as.matrix(r_f)))/100
  }
  return(port_ret)
}

#################
### Main code ###
#################


feats_list = c('LME','BEME','r12_2','OP','Investment','ST_Rev','LT_Rev','AC','IdioVol',"LTurnover")

genTripleSort64=function(feats_list, feat1, feat2, y_min,y_max, data_chunk_path, output_path, factor_path){
  print(feat1)
  print(feat2)
  feats = c('LME', feats_list[feat1], feats_list[feat2])
      
  main_dir = '../data/ts_portfolio/'
  sub_dir = paste(feats[1],'_',feats[2],'_',feats[3],sep='')
  
  dir.create(file.path(output_path, sub_dir), showWarnings = FALSE)
  data_path = paste(data_chunk_path,sub_dir,'/',sep="")
      
  ret_table = triple_sort(data_path, feats[1], feats[2], feats[3], y_min, y_max)
  print(sum(is.na(ret_table)))
  
  ret_table=t(ret_table)
  
  port_ret = remove_rf(ret_table, factor_path)
  port_ret[is.na(port_ret)] = 0
  
  write.table(port_ret, paste(output_path, sub_dir,'/excess_ports.csv', sep=''), sep=',',row.names=F) 
}

