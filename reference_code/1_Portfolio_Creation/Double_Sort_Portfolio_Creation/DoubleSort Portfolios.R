setwd("~/YourWorkingDirectory")
library('dplyr')

double_sort = function(data_path, feat1, feat2, q_num, y_min, y_max){
  ### The function build portfolios and compute the value-averaged returns
  ### Input: data_path: the path for the data
  ###        feat1, feat2: the two features
  ###        q_num: number of cuts based on the quantiles, normally 2 or 5
  ###        y_min, y_max: the range of years
  ret_table = array(0,dim=c(q_num*q_num,(y_max-y_min+1)*12))
  y_time_stamp = 1
  for(y in c(y_min:y_max)){
    print(y)
    data_filenm = paste(data_path,'y',toString(y),'.csv',sep='')
    df_tmp = read.csv(data_filenm)
    ret_table[,c(((y_time_stamp-1)*12+1):(y_time_stamp*12))] = double_sort_helper(df_tmp, feat1, feat2, q_num) 
    y_time_stamp = y_time_stamp+1
  }  
  return(ret_table)
}

double_sort_helper = function(df_tmp, feat1, feat2, q_num){
  ### A helper function that build portfolio for a specific year
  ### Input: df_tmp: the dataframe of a spcific year
  ###        feat1, feat2: two features
  ###        q_num: number of cuts based on the quantiles, normally 2 or 5
  ### Output: ret_tmp: the portfolio returns over a specific year 
  ret_tmp = array(0,dim=c(q_num*q_num,12))
  df_tmp['1'] = 0
  df_tmp['2'] = 0
  for(mon in c(1:12)){
    mask_m = (df_tmp['mm']==mon)
    df_m = df_tmp[mask_m,]
    df_m['1'] = ntile(df_m[feat1], q_num)
    df_m['2'] = ntile(df_m[feat2], q_num)
    for(i in c(1:q_num)){
      for(j in c(1:q_num)){
        mask_port = (df_m['1']==i) & (df_m['2']==j)
        company_val = df_m['size'][mask_port,]
        ret_mon = df_m['ret'][mask_port,]
        ret_tmp[(i-1)*q_num+j,mon] = ret_mon%*%company_val/sum(company_val)
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

for (feat1n in 1:(length(feats_list)-1)){
  for (feat2n in (feat1n+1):length(feats_list)){
    print(feat1n)
    print(feat2n)
    
    feat1 = feats_list[feat1n]
    feat2 = feats_list[feat2n]
    
    q_num = 4 # number of quantile cuts for each feature
    y_min = 1964
    y_max = 2016
    
    main_dir = '../data/ds_portfolio/'
    sub_dir = paste(feat1,'_',feat2,sep='')
    dir.create(file.path(main_dir, sub_dir), showWarnings = FALSE)
    data_path = paste('../data/data_chunk_files_quantile/',sub_dir,'/',sep="")
    
    factor_path = '../data/factor/'
    
    ret_table = double_sort(data_path, feat1, feat2, q_num, y_min, y_max)
    print(sum(is.na(ret_table)))
    
    
    ret_table=t(ret_table)
    
    port_ret = remove_rf(ret_table, factor_path)
    port_ret[is.na(port_ret)] = 0
    
    write.table(port_ret, paste('../data/ds_portfolio/', paste(feat1,feat2,sep = '_'),'/ds_', q_num^2,'excess.csv', sep=''), sep=',',row.names=F) 
  }
}
    
