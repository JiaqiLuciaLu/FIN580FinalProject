library(ggplot2)


SRN_ML_Plot=function(SRFF10Path,plot_path){
  sr = read.table(SRFF10Path, header=T, sep=',')
  graph.width <- 14.7 #16
  graph.height <- 5.7 #9
  text.size <- 16 #20
  
  plot_name = paste(plot_path,"/FF10_Prediction_Methods_Testing_Comparison_ggplot.png", sep="")
  ids = 5:50
  FF_alpha_gg = rbind(data.frame(Id = ids, Type = "AP-Tree (10)", Alpha = as.numeric(sr[1,])),
                      data.frame(Id = ids, Type = "AP-Tree (40)", Alpha = as.numeric(sr[2,])),
                      data.frame(Id = ids, Type = "V-Tree (10)", Alpha = as.numeric(sr[3,])),
                      data.frame(Id = ids, Type = "V-Tree (40)", Alpha = as.numeric(sr[4,])),
                      data.frame(Id = ids, Type = "DL-LS", Alpha = as.numeric(sr[5,])),
                      data.frame(Id = ids, Type = "DL-MV", Alpha = as.numeric(sr[6,])),
                      data.frame(Id = ids, Type = "RF-LS", Alpha = as.numeric(sr[7,])),
                      data.frame(Id = ids, Type = "RF-MV", Alpha = as.numeric(sr[8,])))
  
  g = ggplot(data=FF_alpha_gg, aes(x=Id, y=Alpha, group=Type, shape = Type)) +
    geom_line(aes(linetype=Type, color = Type),size=1.5)+
    geom_point(aes(color=Type), size = 4) + xlab("Number of Portfolios")  + ylab("Testing SR")  +
    labs(color = "Basis portfolios:", linetype = "Basis portfolios:", shape  = "Basis portfolios:") +
    # scale_linetype_manual(values=c("solid", "twodash", "dotted", "longdash"))+
    scale_shape_manual(values = c(16, 17, 15, 4, 11, 8, 7, 2)) +
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