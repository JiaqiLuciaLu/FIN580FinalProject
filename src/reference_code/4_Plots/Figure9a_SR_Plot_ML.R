library(ggplot2)

SR_Plot_ML = function(SR_Summary_File, plot_path){
  graph.width <- 14.7 #16
  graph.height <- 5.7 #9
  text.size <- 16 #20
  
  p = 40
  
  sr = read.table(SR_Summary_File, header=T, sep=',')[1:36,]
  values = sr[order(sr.tree), c(4, 5, 6, 7, 8, 9, 10)]
  
  plot_name = paste(plot_path,"SR_Prediction_Methods_", p, ".png", sep='')
  ids = factor(values$Id, levels = values$Id)
  data.gg = rbind(data.frame(Id = ids, Type = "AP-Tree", Alpha = values[,2]),
                  data.frame(Id = ids, Type = "V-Tree", Alpha = values[,3]),
                  data.frame(Id = ids, Type = "DL-LS", Alpha = values[,4]),
                  data.frame(Id = ids, Type = "DL-MV", Alpha = values[,5]),
                  data.frame(Id = ids, Type = "RF-LS", Alpha = values[,6]),
                  data.frame(Id = ids, Type = "RF-MV", Alpha = values[,7]))
  
  g = ggplot(data=data.gg, aes(x=Id, y=Alpha, group=Type, shape = Type)) +
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

