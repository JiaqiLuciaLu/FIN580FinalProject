library(ggplot2)

SDF_Alpha_plot = function(portN,SDF_Alpha_Path,OrderPath,Plot_Path){
  graph.width <- 14.7 #16
  graph.height <- 5.7 #9
  text.size <- 16 #20
  p = portN
  
  alpha = read.table(SDF_Alpha_Path, header=T, sep=',')
  sr = as.numeric(read.table(OrderPath, header=F, sep=',')[,1])
  
  
  FF_alpha = cbind(alpha[order(sr),c(4, 29, 33, 37)])
  
  plot_name = paste(Plot_Path,"/SDF_tstat_FF3_", p ,".png", sep = '')
  ids = factor(FF_alpha$Id, levels = FF_alpha$Id)
  FF_alpha_gg = rbind(data.frame(Id = ids, Type = "AP-Tree", Alpha = FF_alpha[,4]),
                      data.frame(Id = ids, Type = "Triple Sort (32)", Alpha = FF_alpha[,2]),
                      data.frame(Id = ids, Type = "Triple Sort (64)", Alpha = FF_alpha[,3]))
  
  
  g = ggplot(data=FF_alpha_gg, aes(x=Id, y=Alpha, group=Type, shape = Type)) +
    geom_line(aes(linetype=Type, color = Type),size=1.5)+
    geom_point(aes(color=Type), size = 4) + xlab("Cross-sections")  + ylab("t-stat")  + 
    labs(color = "Basis portfolios:", linetype = "Basis portfolios:", shape  = "Basis portfolios:") +
    theme(text = element_text(size=text.size), 
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
          panel.background = element_blank(), 
          legend.position = "right", #c(0.9, 0.1), 
          legend.text = element_text(size=text.size), 
          axis.line = element_line(colour = "black"),
          axis.text=element_text(size=text.size))
  
  plot(g)
  ggsave(plot_name, g, width = graph.width, height = graph.height)
  
  FF_alpha = cbind(alpha[order(sr),c(4, 30, 34, 38)])
  
  plot_name = paste(Plot_Path,"/SDF_tstat_FF5_", p ,".png", sep = '')
  ids = factor(FF_alpha$Id, levels = FF_alpha$Id)
  FF_alpha_gg = rbind(data.frame(Id = ids, Type = "AP-Tree", Alpha = FF_alpha[,4]),
                      data.frame(Id = ids, Type = "Triple Sort (32)", Alpha = FF_alpha[,2]),
                      data.frame(Id = ids, Type = "Triple Sort (64)", Alpha = FF_alpha[,3]))
  
  
  g = ggplot(data=FF_alpha_gg, aes(x=Id, y=Alpha, group=Type, shape = Type)) +
    geom_line(aes(linetype=Type, color = Type),size=1.5)+
    geom_point(aes(color=Type), size = 4) + xlab("Cross-sections")  + ylab("t-stat")  + 
    labs(color = "Basis portfolios:", linetype = "Basis portfolios:", shape  = "Basis portfolios:") +
    theme(text = element_text(size=text.size), 
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
          panel.background = element_blank(), 
          legend.position = "right", #c(0.9, 0.1), 
          legend.text = element_text(size=text.size), 
          axis.line = element_line(colour = "black"),
          axis.text=element_text(size=text.size))
  
  plot(g)
  ggsave(plot_name, g, width = graph.width, height = graph.height)
  
  
  FF_alpha = cbind(alpha[order(sr),c(4, 31, 35, 39)])
  
  plot_name = paste(Plot_Path,"/SDF_tstat_XSF_", p ,".png", sep = '')
  ids = factor(FF_alpha$Id, levels = FF_alpha$Id)
  FF_alpha_gg = rbind(data.frame(Id = ids, Type = "AP-Tree", Alpha = FF_alpha[,4]),
                      data.frame(Id = ids, Type = "Triple Sort (32)", Alpha = FF_alpha[,2]),
                      data.frame(Id = ids, Type = "Triple Sort (64)", Alpha = FF_alpha[,3]))
  
  
  g = ggplot(data=FF_alpha_gg, aes(x=Id, y=Alpha, group=Type, shape = Type)) +
    geom_line(aes(linetype=Type, color = Type),size=1.5)+
    geom_point(aes(color=Type), size = 4) + xlab("Cross-sections")  + ylab("t-stat")  + 
    labs(color = "Basis portfolios:", linetype = "Basis portfolios:", shape  = "Basis portfolios:") +
    theme(text = element_text(size=text.size), 
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
          panel.background = element_blank(), 
          legend.position = "right", #c(0.9, 0.1), 
          legend.text = element_text(size=text.size), 
          axis.line = element_line(colour = "black"),
          axis.text=element_text(size=text.size))
  
  plot(g)
  ggsave(plot_name, g, width = graph.width, height = graph.height)
  
  
  
  FF_alpha = cbind(alpha[order(sr),c(4, 32, 36, 40)])
  
  plot_name = paste(Plot_Path,"/SDF_tstat_FF11_", p ,".png", sep = '')
  ids = factor(FF_alpha$Id, levels = FF_alpha$Id)
  FF_alpha_gg = rbind(data.frame(Id = ids, Type = "AP-Tree", Alpha = FF_alpha[,4]),
                      data.frame(Id = ids, Type = "Triple Sort (32)", Alpha = FF_alpha[,2]),
                      data.frame(Id = ids, Type = "Triple Sort (64)", Alpha = FF_alpha[,3]))
  
  
  g = ggplot(data=FF_alpha_gg, aes(x=Id, y=Alpha, group=Type, shape = Type)) +
    geom_line(aes(linetype=Type, color = Type),size=1.5)+
    geom_point(aes(color=Type), size = 4) + xlab("Cross-sections")  + ylab("t-stat")  + 
    labs(color = "Basis portfolios:", linetype = "Basis portfolios:", shape  = "Basis portfolios:") +
    theme(text = element_text(size=text.size), 
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
          panel.background = element_blank(), 
          legend.position = "right", #c(0.9, 0.1), 
          legend.text = element_text(size=text.size), 
          axis.line = element_line(colour = "black"),
          axis.text=element_text(size=text.size))
  
  plot(g)
  ggsave(plot_name, g, width = graph.width, height = graph.height)
  
  
  
  
}


