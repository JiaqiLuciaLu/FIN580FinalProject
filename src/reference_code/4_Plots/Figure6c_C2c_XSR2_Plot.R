library(ggplot2)

XSR2_plot=function(R2TreePath, R2TSPath, R2TS64Path, OrderPath){
  p = 40
  
  sr.tree = as.numeric(read.table(OrderPath, header=F, sep=',')[,1])
  r2.tree = read.table(R2TreePath, header=T, sep=',')[1:36,]
  r2.ts = read.table(R2TSPath, header=T, sep=',')[1:36,]
  r2.ts64 = read.table(R2TS64Path, header=T, sep=',')[1:36,]
  
  
  graph.width <- 14.7 #16
  graph.height <- 5.7 #9
  text.size <- 16 #20
  
  values.tree <- r2.tree[order(sr.tree),c(4,10, 12, 14, 16)]
  values.ts <- r2.ts[order(sr.tree),c(4,10,12, 14, 16)]
  values.ts64 <- r2.ts64[order(sr.tree),c(4,10,12, 14, 16)]
  
  ####################  R2 wrt FF3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  plot_name = paste("../../Data/plot/R2_FF3_adj_", p, ".png", sep="")
  ids = factor(values.tree$Id, levels = values.tree$Id)
  data.gg = rbind(data.frame(Id = ids, Type = "AP-Tree", Alpha = values.tree[,2]),
                  data.frame(Id = ids, Type = "Triple Sort (32)", Alpha = values.ts[,2]),
                  data.frame(Id = ids, Type = "Triple Sort (64)", Alpha = values.ts64[,2])
  )
  
  g = ggplot(data=data.gg, aes(x=Id, y=Alpha, group=Type, shape = Type)) +
    geom_line(aes(linetype=Type, color = Type), size=1.5)+
    geom_point(aes(color=Type), size = 4) + xlab("Cross-sections")  + ylab("R2-Adjusted")  + 
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
  
  ####################  R2 wrt FF5 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  plot_name = paste("../../Data/plot/R2_FF5_adj_", p, ".png", sep="")
  ids = factor(values.tree$Id, levels = values.tree$Id)
  data.gg = rbind(data.frame(Id = ids, Type = "AP-Tree", Alpha = values.tree[,3]),
                  data.frame(Id = ids, Type = "Triple Sort (32)", Alpha = values.ts[,3]),
                  data.frame(Id = ids, Type = "Triple Sort (64)", Alpha = values.ts64[,3])
  )
  
  g = ggplot(data=data.gg, aes(x=Id, y=Alpha, group=Type, shape = Type)) +
    geom_line(aes(linetype=Type, color = Type), size=1.5)+
    geom_point(aes(color=Type), size = 4) + xlab("Cross-sections")  + ylab("R2-Adjusted")  + 
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
  
  
  ####################  R2 wrt XSF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  plot_name = paste("../../Data/plot/R2_XSF_adj_", p, ".png", sep="")
  ids = factor(values.tree$Id, levels = values.tree$Id)
  data.gg = rbind(data.frame(Id = ids, Type = "AP-Tree", Alpha = values.tree[,4]),
                  data.frame(Id = ids, Type = "Triple Sort (32)", Alpha = values.ts[,4]),
                  data.frame(Id = ids, Type = "Triple Sort (64)", Alpha = values.ts64[,4])
  )
  
  g = ggplot(data=data.gg, aes(x=Id, y=Alpha, group=Type, shape = Type)) +
    geom_line(aes(linetype=Type, color = Type), size=1.5)+
    geom_point(aes(color=Type), size = 4) + xlab("Cross-sections")  + ylab("R2-Adjusted")  + 
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
  
  
  ####################  R2 wrt FF11 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  plot_name = paste("../../Data/plot/R2_FF11_adj_", p, ".png", sep="")
  ids = factor(values.tree$Id, levels = values.tree$Id)
  data.gg = rbind(data.frame(Id = ids, Type = "AP-Tree", Alpha = values.tree[,5]),
                  data.frame(Id = ids, Type = "Triple Sort (32)", Alpha = values.ts[,5]),
                  data.frame(Id = ids, Type = "Triple Sort (64)", Alpha = values.ts64[,5])
  )
  
  g = ggplot(data=data.gg, aes(x=Id, y=Alpha, group=Type, shape = Type)) +
    geom_line(aes(linetype=Type, color = Type), size=1.5)+
    geom_point(aes(color=Type), size = 4) + xlab("Cross-sections")  + ylab("R2-Adjusted")  + 
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


