library(ggplot2)

SRN_Plot=function(GridSearchPath,plot_path){
  graph.width <- 14
  graph.height <- 9
  text.size <- 24
  
  ids = 5:50
  
  feat1 = 4
  feat2 = 5
  
  filename = paste(c('LME', feats_list[feat1], feats_list[feat2]),collapse = '_')
  
  sr_path = paste(GridSearchPath,filename,'/SR_N.csv',sep='')
  sr = read.table(sr_path, header=T, sep=',')
  
  
  data.gg = data.frame(Id = ids, Type = 'Validation', srs = as.numeric(sr[2,]))
  
  plot_name = paste(plot_path,filename, "_Validation_gg.png", sep="")
  
  g = ggplot(data=data.gg, aes(x=Id, y=srs, group=Type)) +
    geom_line(aes(color=Type), color = 'Red', size = 1.5)+
    geom_point(aes(color=Type), color = 'Red') + xlab("Number of Portfolios")  + ylab("Validation SR")  + 
    theme(text = element_text(size=text.size), 
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
          panel.background = element_blank(), 
          legend.position = c(0.9, 0.1), 
          legend.text = element_text(size=text.size), 
          axis.line = element_line(colour = "black"),
          axis.text=element_text(size=text.size))
  
  plot(g)
  ggsave(plot_name, g, width = graph.width, height = graph.height)
  
  data.gg = data.frame(Id = ids, Type = 'Testing', srs = as.numeric(sr[2,]))
  
  plot_name = paste(plot_path,filename, "_Testing_gg.png", sep="")
  
  
  g = ggplot(data=data.gg, aes(x=Id, y=srs, group=Type)) +
    geom_line(aes(color=Type), color = 'Blue', size = 1.5)+
    geom_point(aes(color=Type), color = 'Blue') + xlab("Number of Portfolios")  + ylab("Testing SR")  + 
    theme(text = element_text(size=text.size), 
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
          panel.background = element_blank(), 
          legend.position = c(0.9, 0.1), 
          legend.text = element_text(size=text.size), 
          axis.line = element_line(colour = "black"),
          axis.text=element_text(size=text.size))
  
  plot(g)
  ggsave(plot_name, g, width = graph.width, height = graph.height)
}


      
      
      