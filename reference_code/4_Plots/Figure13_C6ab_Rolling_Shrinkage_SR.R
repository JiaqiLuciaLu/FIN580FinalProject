library(ggplot2)



SR_Plot_Rolling_Shrinkage = function(SR_Summary_File, plot_path,plotfilename, ordercolid,linenames,lineids){
  graph.width <- 14.7 #16
  graph.height <- 5.7 #9
  text.size <- 16 #20
  
  rolling_sr = read.table(SR_Summary_File, header=T, sep=',')
  sr = rolling_sr[,ordercolid]
  
  ids = factor(rolling_sr$Id[order(sr)], levels = rolling_sr$Id[order(sr)])
  
  rolling_sr = cbind(rolling_sr[order(sr),])
  
  plot_name = paste(plot_path,filename,plotfilename, sep = '')
  SR_gg = rbind(data.frame(Id = ids, Type = linenames[1], SR = rolling_sr[,lineids[1]]),
                data.frame(Id = ids, Type = linenames[2], SR = rolling_sr[,lineids[2]]),
                data.frame(Id = ids, Type = linenames[3], SR = rolling_sr[,lineids[3]]),
                data.frame(Id = ids, Type = linenames[4], SR = rolling_sr[,lineids[4]]))
  
  
  g = ggplot(data=SR_gg, aes(x=Id, y=SR, group=Type, shape = Type)) +
    geom_line(aes(linetype=Type, color = Type), size=1.5)+
    geom_point(aes(color=Type), size = 4) + xlab("Cross-sections")  + ylab("Monthly Sharpe Ratio (SR)")  + 
    labs(color = "Basis portfolios:", linetype = "Basis portfolios:", shape  = "Basis portfolios:") +
    # scale_linetype_manual(values=c("solid", "twodash", "dotted", "longdash"))+
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