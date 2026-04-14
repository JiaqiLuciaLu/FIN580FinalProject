# rm(list = ls())
# setwd("~/YourWorkingDirectory")


filter = c("X1111", "X2222", "X3333")

filterTreePorts=function(feats_list, feat1,feat2,tree_portfolio_path){
  feats_chosen = c('LME',feats_list[feat1],feats_list[feat2])
  sub_dir = paste(feats_chosen,collapse = '_')
  
  ports_path = paste(tree_portfolio_path, sub_dir, '/', sep='')
  port_ret = read.table(paste(ports_path, 'level_all_excess_combined.csv', sep=''), header=T, sep=',')
  
  filt = (substring(colnames(port_ret), 1, 5) %in% filter) & (sapply(colnames(port_ret), nchar) == 11)
  port_ret = port_ret[,!filt]
  
  for (f in 1:3){
    f_min = read.table(paste(ports_path, 'level_all_',feats_chosen[f],'_min.csv', sep=''), header=T, sep=',')
    f_max = read.table(paste(ports_path, 'level_all_',feats_chosen[f],'_max.csv', sep=''), header=T, sep=',')
    f_min = f_min[,!filt]
    f_max = f_max[,!filt]
    write.csv(f_min, paste(ports_path, 'level_all_',feats_chosen[f],'_min_filtered.csv', sep=''), row.names = F)
    write.csv(f_max, paste(ports_path, 'level_all_',feats_chosen[f],'_max_filtered.csv', sep=''), row.names = F)
  }
  write.csv(port_ret, paste(ports_path, 'level_all_excess_combined_filtered.csv', sep=''), row.names = F)
}
