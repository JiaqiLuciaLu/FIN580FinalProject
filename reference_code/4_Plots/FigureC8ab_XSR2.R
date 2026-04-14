## Source file to compute the statistics of the portfolios

### The basic input "port_ret" is always a T*N matrix, N is the number of portfolios and
### T is the length of time span. 
# rm(list = ls())
require(foreign)
require(plm)
require(lmtest)
require(ggplot2)


FF_regression = function(port_ret, factor_path, option, T0, T1){
  ### Fama French regression
  ### First to choose the factor
  if(class(option)=='numeric'){
    # portfolio specific FF regression
    factor_file = paste(factor_path,'tradable_factors.csv',sep='')
    factor_mat = read.table(factor_file, header=T, sep=',')
    factor = factor_mat[T0:T1,option]
    X = as.matrix(cbind(rep(1,nrow(factor)),factor))
  }
  else if(option=='FF3'){
    # single factor
    factor_file = paste(factor_path,'tradable_factors.csv',sep='')
    factor_mat = read.table(factor_file, header=T, sep=',')
    factor = factor_mat[T0:T1,2:4]
    X = as.matrix(cbind(rep(1,length(factor)),factor))
    factor = matrix(factor, nrow=nrow(X))
  }else if(option=='FF5'){
    # FF3
    factor_file = paste(factor_path,'tradable_factors.csv',sep='')
    factor_mat = read.table(factor_file, header=T, sep=',')
    factor = factor_mat[T0:T1,c(2,3,4,6,7)]
    X = as.matrix(cbind(rep(1,nrow(factor)),factor))
  }else if(option=='FF11'){
    # FF 5
    factor_file = paste(factor_path,'tradable_factors.csv',sep='')
    factor_mat = read.table(factor_file, header=T, sep=',')
    factor = factor_mat[T0:T1,2:12]
    X = as.matrix(cbind(rep(1,nrow(factor)),factor))
  }
  
  
  model = lm(port_ret[T0:T1,1]~X-1)
  avg_ret = mean(port_ret[T0:T1,1])
  alphas = coef(summary(model))[1,1]
  alphas_se = coef(summary(model))[1,2]
  betas = coef(summary(model))[-1,1]
  pred_ret = sum(betas*colMeans(X)[-1])
  eps = port_ret[T0:T1,1] - as.numeric(betas%*%t(X[,-1]))
  
  coeff_ = coef(summary(model))[,1]
  se_ = coef(summary(model))[,2]
  t_ = coef(summary(model))[,3]
  p_ = coef(summary(model))[,4]
  rs_= summary(model)$r.squared
  adj_rs_ = summary(model)$adj.r.squared
  for (i in 2:ncol(port_ret[T0:T1,])){
    model = lm(port_ret[T0:T1,i]~X-1)
    avg_ret[i] = mean(port_ret[T0:T1,i])
    betas = rbind(betas,coef(summary(model))[-1,1])
    alphas[i] = coef(summary(model))[1,1]
    pred_ret[i] = sum(betas[nrow(betas),]*colMeans(X)[-1])
    eps = cbind(eps, port_ret[T0:T1,i] - as.numeric(betas[i,]%*%t(X[,-1])))
    
    coeff_ = cbind(coeff_, coef(summary(model))[,1])
    se_ = cbind(se_, coef(summary(model))[,2])
    t_ = cbind(t_, coef(summary(model))[,3])
    p_ = cbind(p_, coef(summary(model))[,4])
    rs_= c(rs_, summary(model)$r.squared)
    adj_rs_ = c(adj_rs_, summary(model)$adj.r.squared)
  }
  
  
  rs = 1 - sum(alphas^2)/sum(avg_ret^2)
  rsq = c(rs,1-(1-rs)*(length(pred_ret))/(length(pred_ret)-ncol(X)), 1 - sum(eps^2)/sum(port_ret[T0:T1,]^2))
  
  
  return(list(coeff_,se_, t_, p_,rs_,adj_rs_,rsq))
}

compute_Statistics = function(port_ret, factor_path, option, T0, T1){
  ### Compute:
  ### Mean Absolute Alpha
  ### Maximum Sharpe
  ### GRS
  
  mean_alpha = rep(0,4)
  cs_rsq = rep(0,4)
  cs_adjrsq = rep(0,4)
  cs_r = rep(0,4)
  
  
  ### F1 regression
  res_FF = FF_regression(port_ret, factor_path, 'FF3', T0, T1)
  mean_alpha[1] = mean(abs(res_FF[[1]][1,]))
  alpha1 = res_FF[[1]][1,]
  se1 = res_FF[[2]][1,]
  t1 = res_FF[[3]][1,]
  p1 = res_FF[[4]][1,]
  
  rs1 = res_FF[[5]]
  adjrs1 = res_FF[[6]]
  cs_rsq[1] = res_FF[[7]][1]
  cs_adjrsq[1] = res_FF[[7]][2]
  cs_r[1] = res_FF[[7]][3]
  ### F3 regression
  res_FF = FF_regression(port_ret, factor_path, 'FF5', T0, T1)
  mean_alpha[2] = mean(abs(res_FF[[1]][1,]))
  alpha2 = res_FF[[1]][1,]
  se2 = res_FF[[2]][1,]
  t2 = res_FF[[3]][1,]
  p2 = res_FF[[4]][1,]
  rs2 = res_FF[[5]]
  adjrs2 = res_FF[[6]]
  cs_rsq[2] = res_FF[[7]][1]
  cs_adjrsq[2] = res_FF[[7]][2]
  cs_r[2] = res_FF[[7]][3]
  ### F5 regression
  res_FF = FF_regression(port_ret, factor_path, option, T0, T1)
  mean_alpha[3] = mean(abs(res_FF[[1]][1,]))
  alpha3 = res_FF[[1]][1,]
  se3 = res_FF[[2]][1,]
  t3 = res_FF[[3]][1,]
  p3 = res_FF[[4]][1,]
  rs3 = res_FF[[5]]
  adjrs3 = res_FF[[6]]
  cs_rsq[3] = res_FF[[7]][1]
  cs_adjrsq[3] = res_FF[[7]][2]
  cs_r[3] = res_FF[[7]][3]
  ### F3 regression
  res_FF = FF_regression(port_ret, factor_path, 'FF11', T0, T1)
  mean_alpha[4] = mean(abs(res_FF[[1]][1,]))
  alpha4 = res_FF[[1]][1,]
  se4 = res_FF[[2]][1,]
  t4 = res_FF[[3]][1,]
  p4 = res_FF[[4]][1,]
  rs4 = res_FF[[5]]
  adjrs4 = res_FF[[6]]
  cs_rsq[4] = res_FF[[7]][1]
  cs_adjrsq[4] = res_FF[[7]][2]
  cs_r[4] = res_FF[[7]][3]
  return(list(list(alpha1, alpha2, alpha3, alpha4), c(mean_alpha, cs_rsq,cs_adjrsq, cs_r), 
              list(se1,se2,se3,se4), list(t1,t2,t3,t4), list(p1,p2,p3,p4), 
              list(rs1,rs2,rs3,rs4), list(adjrs1,adjrs2,adjrs3,adjrs4)))
}

#################
### Main code ###
#################

XSR2=function(feats_list, feat1, feat2, factor_path, port_path, port_name, plot_path_base, port_type){
  factors = c('Date','market',feats_list)
  T0 = 361
  T1 = 636
  
  feats_chosen = c('LME',feats_list[feat1],feats_list[feat2])
  print(feats_chosen)
  option = as.numeric(match(c('market',feats_chosen), factors))
  sub_dir = paste(feats_chosen,collapse = '_')
  
  port_ret = read.table(paste(port_path, sub_dir,port_name, sep=''), header=T, sep=',')
  result = compute_Statistics(port_ret, factor_path, option, T0, T1)
  
  alphas = rbind(result[[1]][[1]],result[[1]][[2]],result[[1]][[3]],result[[1]][[4]])
  ses = rbind(result[[3]][[1]],result[[3]][[2]],result[[3]][[3]],result[[3]][[4]])
  ts = rbind(result[[4]][[1]],result[[4]][[2]],result[[4]][[3]],result[[4]][[4]])
  ps = rbind(result[[5]][[1]],result[[5]][[2]],result[[5]][[3]],result[[5]][[4]])
  rs = rbind(result[[6]][[1]],result[[6]][[2]],result[[6]][[3]],result[[6]][[4]],
             result[[7]][[1]],result[[7]][[2]],result[[7]][[3]],result[[7]][[4]])
  
  if (port_type=='ts32'){
    cnames =  c("111","112","113","114","121","122","123","124","131",
     "132","133","134","141","142","143","144",
     "211","212","213","214","221","222","223","224","231",
     "232","233","234","241","242","243","244")
  }else if (port_type=='ts64'){
    cnames =  c("111","112","113","114","121","122","123","124","131",
                "132","133","134","141","142","143","144",
                "211","212","213","214","221","222","223","224","231",
                "232","233","234","241","242","243","244",
                "311","312","313","314","321","322","323","324","331",
                "332","333","334","341","342","343","344",
                "411","412","413","414","421","422","423","424","431",
                "432","433","434","441","442","443","444")
  }else{
    cnames = colnames(port_ret)
    cnames = substring(cnames, 2, nchar(cnames))
  }
  colnames(alphas) = cnames

  ids = colnames(alphas)
  ids = factor(ids, levels = ids)
  colnames(rs) = cnames
  
  plot_path = paste(plot_path_base,port_type,'/TimeSeriesAlpha/',sub_dir,'/',sep='')
  dir.create(paste(plot_path_base,port_type,sep=''), showWarnings = FALSE)
  dir.create(paste(plot_path_base,port_type,'/TimeSeriesAlpha',sep=''), showWarnings = FALSE)
  dir.create(file.path(plot_path), showWarnings = FALSE)
  
  factor_group_names = c('FF3', 'FF5', 'XSF', 'FF11')
  text.size = 15
  for (i in 1:4){
    plot_name = paste(plot_path, 'TimeSeriesAlpha', '_', factor_group_names[i],'.png', sep='')
    
    df = data.frame(id = ids, 
                    min3 = alphas[i,] - 3 * ses[i,], 
                    min = alphas[i,] - 2 * ses[i,], 
                    lower = alphas[i,] - ses[i,], 
                    med = alphas[i,], 
                    upper = alphas[i,] + ses[i,], 
                    max = alphas[i,] + 2 * ses[i,], 
                    max3 = alphas[i,] + 3 * ses[i,])
    
    g = ggplot(df, aes(x = id, ymin=min, lower=lower, 
                       middle=med, upper=upper, ymax=max)) +
      geom_boxplot(stat="identity")  + 
      geom_line(data = df, aes(x = id, y = min3, group = 1), color = "blue",linetype="dotted") + 
      geom_hline(aes(yintercept=0), color="red") + 
      geom_line(data = df, aes(x = id, y = max3, group = 1), color = "blue",linetype="dotted") + 
      #ggtitle(paste("Time-series pricing error of ", paste(feats_chosen, collapse = "-"), " Triple Sorting portfolios on ", factor_group_names[i]," factors", sep = "")) +
      xlab("Portfolio") + 
      ylab("Pricing Error") +
      theme(text = element_text(size=text.size, face = "bold"), 
            panel.grid.major = element_blank(), 
            panel.grid.minor = element_blank(),
            panel.background = element_blank(), 
            legend.position = c(0.9, 0.1), 
            legend.text = element_text(size=text.size), 
            axis.line = element_line(colour = "black"),
            axis.text=element_text(size=text.size, face = "bold"),
            axis.text.x = element_text(angle = 90, hjust = 1))
    
    ggsave(plot_name, g, width = 16, height = 6)
  }
  
  alphas = rbind(alphas,ses,ts,ps)  
  stats_path = paste(port_path,'/XSR2Tests/',sub_dir,'/',sep='')
  dir.create(paste(port_path,'/XSR2Tests',sep=''), showWarnings = FALSE)
  dir.create(file.path(stats_path), showWarnings = FALSE)
  
  write.table(alphas, paste(stats_path,'alpha.csv', sep=''), sep=',',row.names=F) 
  write.table(result[[2]], paste(stats_path,'R2.csv', sep=''), sep=',',row.names=F) 
}

