# Generate value-weighted returns and min/max of characteristics of tree portfolios for one tree
tree_portfolio = function(data_path, feat_list, tree_depth, q_num, y_min, y_max, file_prefix, feats){
  n_feats = length(feats)
  ret_table = array(0,dim=c((y_max-y_min+1)*12,q_num^(tree_depth+1)-1))
  feat_min_table = list()
  feat_max_table = list()
  for (f in 1:n_feats){
    feat_min_table[[f]] = array(0,dim=c((y_max-y_min+1)*12,q_num^(tree_depth+1)-1))
    feat_max_table[[f]] = array(0,dim=c((y_max-y_min+1)*12,q_num^(tree_depth+1)-1))
  }
  
  for(y in c(y_min:y_max)){
    if(y%%5==0){print(y)}
    data_filenm = paste(data_path,file_prefix,toString(y),'.csv',sep='')
    df_m = read.csv(data_filenm)
    df_m = tree_portfolio_y(df_m, feat_list, tree_depth, q_num)
    
    for (i in 1:tree_depth){
      for(k in c(1:i)){
        df_m[paste('port',i,sep='')] = df_m[paste('port',i,sep='')] + (df_m[toString(k)]-1)*(q_num^(i-k))
      }
    }
    
    for (i in 0:tree_depth){
      for(m in c(1:12)){
        for(k in c(1:q_num^i)){
          mask_port = (df_m['mm']==m)&(df_m[paste('port',i,sep='')]==k)
          company_val = df_m['size'][mask_port]
          ret_mon = df_m['ret'][mask_port]
          feat_m = list()
          
          ret_table[12*(y-y_min)+m,2^i-1+k] = ret_mon%*%company_val/sum(company_val)
          
          for (f in 1:n_feats){
            feat_m[[f]] = df_m[feats[f]][mask_port]
            feat_min_table[[f]][12*(y-y_min)+m,2^i-1+k] = min(feat_m[[f]])
            feat_max_table[[f]][12*(y-y_min)+m,2^i-1+k] = max(feat_m[[f]])
          }
        }
      }
    }
  }  
  ret_list = list()
  ret_list[[1]]=ret_table
  for (f in 1:n_feats){
    ret_list[[2*f]]=feat_min_table[[f]]
    ret_list[[2*f+1]]=feat_max_table[[f]]
  }
  return(ret_list)
}

tree_portfolio_y = function(df_tmp, feat_list, tree_depth, q_num){
  for(k in c(1:tree_depth)){
    df_tmp[toString(k)] = 0
  }
  for (i in 0:tree_depth){
    df_tmp[paste('port',i,sep='')] = 1
  }
  
  for(m in c(1:12)){
    mask_m = (df_tmp['mm']==m)
    df_m = df_tmp[mask_m,]
    df_m[toString(1)] = ntile(df_m[feat_list[1]], q_num)
    for(val in c(1:q_num)){
      k = 1
      mask_m_tmp = (df_m[toString(1)] == val)
      df_m_recurse = df_m[mask_m_tmp,]
      df_m[mask_m_tmp,] = tree_portfolio_y_helper(df_m_recurse, feat_list, k, tree_depth, q_num)
    }
    df_tmp[mask_m,] = df_m
  }
  return(df_tmp)
}

tree_portfolio_y_helper = function(df_m_recurse, feat_list, k, tree_depth, q_num){
  #print(k)
  k = k+1
  df_m_recurse[toString(k)] = ntile(df_m_recurse[feat_list[k]], q_num)
  if(k<tree_depth){
    for(val in c(1:q_num)){
      mask_recurse = (df_m_recurse[toString(k)]==val)
      df_m_recurse[mask_recurse,] = tree_portfolio_y_helper(df_m_recurse[mask_recurse,], 
                                                            feat_list, k, tree_depth, q_num)
    }
  }
  
  return(df_m_recurse)
}
