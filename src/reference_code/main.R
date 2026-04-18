# This code is tested under the following package versions:
# rstudioapi: 0.15.0
# lars: 1.3
# ggplot2: 3.4.4
# dplyr: 1.1.4
# doSNOW: 1.0.20
# foreach: 1.5.2

list.of.packages <- c("rstudioapi","lars","ggplot2","dplyr","doSNOW","foreach")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)


rm(list = ls())
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

feats_list = c('LME','BEME','r12_2','OP','Investment','ST_Rev','LT_Rev','AC','IdioVol',"LTurnover")
feat1 = 4 # Operating Prof
feat2 = 5 # Investment
y_min = 1964
y_max = 2016
n_train_valid = 360 # n months for training and validation
cvN = 3 #cross-validation counts

portN=10

#Change this to FALSE for max compatibility
# RunParallel = TRUE
# pralleln = 6
RunParallel = FALSE
pralleln = 1

raw_data_path = '../Data/ret_characteristics/'
data_chunk_path = '../Data/data_chunk_files_quantile/'
tree_portfolio_path = '../Data/tree_portfolio_quantile/'
ts32_path = '../Data/ts_portfolio/'
ts64_path = '../Data/ts64_portfolio/'
tree_grid_search_path = '../Data/TreeGridSearch/'
ts32_grid_search_path = '../Data/TSGridSearch/'
ts64_grid_search_path = '../Data/TS64GridSearch/'
factor_path = '../Data/factor/'
plot_path = '../Data/plot/'
dir.create(data_chunk_path, showWarnings = FALSE)
dir.create(tree_portfolio_path)
dir.create(ts32_path)
dir.create(ts64_path)
dir.create(tree_grid_search_path)
dir.create(ts32_grid_search_path)
dir.create(ts64_grid_search_path)
dir.create(plot_path)
OrderPath = '../../Data/Summary/SR.csv'

#####################################################################
################  Portfolio Generation from Raw Data ################
#####################################################################

# This is the code used to convert RAW data downloaded from CRSP/Compustat to Yearly files with only
# return, size (for value weighting), and corresponding quantile characteristics
# To comply with data sharing license from WRDS, we do not share this raw data but only share the
# outputs from this step, note that we have also added random noise to return and size so this data
# is only for illustration purpose

# source('1_Portfolio_Creation/Tree_Portfolio_Creation/Step1_Combine_Raw_Chars_Convert_Quantile_Split_Yearly_Chunks.R')
#create_yearly_chunks(y_min,y_max,feats_list,feat1,feat2,input_path,output_path,add_noise=FALSE)
# create_yearly_chunks(y_min,y_max,feats_list,feat1,feat2,raw_data_path,data_chunk_path,TRUE)
  

# Steps to generate the tree portfolios
source('1_Portfolio_Creation/Tree_Portfolio_Creation/Step2_Generate_Tree_Portfolios_All_Levels_Char_Minmax.R')
tree_depth = 4
#create_tree_portfolio(y_min,y_max,tree_depth,feats_list,feat1,feat2,input_path,output_path,runparallel,pralleln)
create_tree_portfolio(y_min,y_max,tree_depth,feats_list,feat1,feat2,data_chunk_path,tree_portfolio_path,RunParallel,pralleln)
source('1_Portfolio_Creation/Tree_Portfolio_Creation/Step3_RmRf_Combine_Trees.R')
# combinetrees(feats_list, feat1, feat2, tree_depth, factor_path, tree_sort_path_base)
combinetrees(feats_list, feat1, feat2, tree_depth, factor_path, tree_portfolio_path)
source('1_Portfolio_Creation/Tree_Portfolio_Creation/Step4_Filter_SingleSorted_Tree_Ports.R')
# filterTreePorts(feats_list, feat1,feat2,tree_portfolio_path)
filterTreePorts(feats_list,feat1,feat2,tree_portfolio_path)

# Steps to generate the triple sort 32 portfolios
source('1_Portfolio_Creation/Triple_Sort_Portfolio_Creation/TripleSort32_Portfolios.R')
#genTripleSort32(feats_list, feat1, feat2, y_min,y_max, data_chunk_path, output_path, factor_path)
genTripleSort32(feats_list, feat1, feat2, y_min,y_max, data_chunk_path, ts32_path, factor_path)

# Steps to generate the triple sort 64 portfolios
source('1_Portfolio_Creation/Triple_Sort_Portfolio_Creation/TripleSort64_Portfolios.R')
#genTripleSort32(feats_list, feat1, feat2, y_min,y_max, data_chunk_path, output_path, factor_path)
genTripleSort64(feats_list, feat1, feat2, y_min, y_max, data_chunk_path, ts64_path, factor_path)


#################################################################################
################  AP Pruning on Base Portfolios with Grid Search ################
#################################################################################


# In the paper we used a much larger grid search on the following parameter, but doing the full parameter search can take a long time to finish
# lambda0 = seq(0, 0.9, 0.05)
# lambda2 = 0.1^seq(5, 8, 0.25)

# For demonstration, we put the below smaller grid of hyper-parameters
lambda0 = seq(0.5, 0.6, 0.05)
lambda2 = 0.1^seq(7, 7.5, 0.25)

source('2_AP_Pruning/AP_Pruning.R')
# AP_Pruning(feats_list, feat1, feat2, input_path,input_file_name, output_path,n_train_valid,cvN,runFullCV,kmax, RunParallel, ParallelN,IsTree,lambda0, lambda2)
# Due to dimension of trees, the grid search on trees can take a while without parallel computing
AP_Pruning(feats_list, feat1, feat2, tree_portfolio_path,'/level_all_excess_combined_filtered.csv',tree_grid_search_path,n_train_valid,cvN,FALSE,50, RunParallel, pralleln, TRUE,lambda0, lambda2)
AP_Pruning(feats_list, feat1, feat2, ts32_path,'/excess_ports.csv',ts32_grid_search_path,n_train_valid,cvN, TRUE, 32, RunParallel, pralleln, FALSE,lambda0, lambda2)
AP_Pruning(feats_list, feat1, feat2, ts64_path,'/excess_ports.csv',ts64_grid_search_path,n_train_valid,cvN, TRUE, 64, RunParallel, pralleln, FALSE,lambda0, lambda2)


#################################################################################
#############  Parameter Search and Asset Pricing Regression Tests ##############
#################################################################################

# Grid Search on Lambda tuning parameters
source('3_Metrics_Collection/Pick_Best_Lambda.R')
# pickBestLambda(feats_list, feat1, feat2, ap_prune_result_path,portN,lambda0,lambda2, portfolio_path, port_name, fullCV=FALSE, writetable=TRUE)
pickBestLambda(feats_list, feat1, feat2, tree_grid_search_path,portN,lambda0,lambda2, tree_portfolio_path,'/level_all_excess_combined_filtered.csv')
pickBestLambda(feats_list, feat1, feat2, ts32_grid_search_path,portN,lambda0,lambda2, ts32_path,'/excess_ports.csv')
pickBestLambda(feats_list, feat1, feat2, ts64_grid_search_path,portN,lambda0,lambda2, ts64_path,'/excess_ports.csv')
pickBestLambda(feats_list, feat1, feat2, ts32_grid_search_path,32,lambda0,lambda2, ts32_path,'/excess_ports.csv')
pickBestLambda(feats_list, feat1, feat2, ts64_grid_search_path,64,lambda0,lambda2, ts64_path,'/excess_ports.csv')

# Time Series Regression tests wrt factors. The results are collected to form table 1 and 2
# Re-run this on the sample where we remove small cap stocks gives table 3
source('3_Metrics_Collection/SDF_TimeSeries_Regressions.R')
# SDF_regression(feats_list, feat1, feat2, factor_path, port_path, port_name,weight_name)
SDF_regression(feats_list, feat1, feat2, factor_path, tree_grid_search_path, '/Selected_Ports_10.csv','/Selected_Ports_Weights_10.csv')
SDF_regression(feats_list, feat1, feat2, factor_path, ts32_grid_search_path, '/Selected_Ports_10.csv','/Selected_Ports_Weights_10.csv')
SDF_regression(feats_list, feat1, feat2, factor_path, ts64_grid_search_path, '/Selected_Ports_10.csv','/Selected_Ports_Weights_10.csv')

# Collect Sharpe Ratio results by different number of portfolios
source('3_Metrics_Collection/SR_N.R')
# pickSRN(feats_list,feat1,feat2,grid_search_path,mink,maxk,lambda0,lambda2,port_path,port_file_name)
pickSRN(feats_list,feat1,feat2,tree_grid_search_path,5,50,lambda0,lambda2,tree_portfolio_path,'/level_all_excess_combined_filtered.csv')

#################################################################################
####################################  Plots  ####################################
#################################################################################


### Used to generate the base empirical bounds for Figure 1, actual code to make the plots is in Python
source('1_Portfolio_Creation/Tree_Portfolio_Creation/Generate_2Char_Tree_Portfolios_All_Levels_Char_Minmax.R')
create_tree_portfolio(y_min,y_max,tree_depth,feats_list,feat1,feat2,data_chunk_path,tree_portfolio_path,FALSE,0)
source('1_Portfolio_Creation/Tree_Portfolio_Creation/Combine_2Char_Trees.R')
combinetrees(feats_list, feat1, tree_depth, factor_path, tree_portfolio_path)


# This part requires running all 36 combinations and collect the out of sample best Sharpe Ratios (by validation performance)
# in a single csv file for plotting
source('4_Plots/Figure6a_7_8_SR_Plot_XSF.R')
# SR_Plot_XSF(SR_Summary_File, plot_path)
# SR_Plot_XSF('../../Data/SRSummary/SR.csv', '../../Data/plot/')

# This part requires running all 36 combinations and collect the SDF Alpha's t-test results into a single file
source('4_Plots/Figure6b_C2ab_C4_C5_SDF_AlphaT.R')
dir.create(paste(plot_path,'/SDF_Alpha/',sep=''))
# SDF_Alpha_plot(portN,SDF_Alpha_Path,OrderPath,Plot_Path)
# SDF_Alpha_plot(portN,SDF_Alpha_Path,OrderPath,paste(plot_path,'/SDF_Alpha/',sep=''))

# This part requires running all 36 combinations and collect the XSR2 results into a single csv file for plotting
source('4_Plots/Figure6c_C2c_XSR2_Plot.R')
R2TreePath = '../../Data/Summary/XSR2CombinedTree.csv'
R2TSPath = '../../Data/Summary/XSR2CombinedTS.csv'
R2TS64Path = '../../Data/Summary/XSR2CombinedTS64.csv'
# XSR2_plot(R2TreePath, R2TSPath, R2TS64Path, OrderPath)
# XSR2_plot(R2TreePath, R2TSPath, R2TS64Path, OrderPath)

# This part requires running all 36 combinations and collect the sharpe results into a single csv file for plotting
source('4_Plots/Figure9a_SR_Plot_ML.R')
# SR_Plot_ML(SR_Summary_File, plot_path)
# SR_Plot_ML(SR_Summary_File, plot_path)

# This part requires running all 36 combinations and collect the Alpha t-stat results into a single csv file for plotting
source('4_Plots/Figure9b_SDF_Alpha_ML.R')
# SDF_Alpha_plot_ML(portN,SDF_Alpha_Path,OrderPath,Plot_Path)
# SDF_Alpha_plot_ML(portN,SDF_Alpha_Path_ML,OrderPath,paste(plot_path,'/SDF_Alpha/',sep=''))

### Used to generate impact of portfolio number on sharpe ratios
source('4_Plots/Figure10ac_SDF_SR_N.R')
# SRN_Plot(GridSearchPath,plot_path)
dir.create(paste(plot_path,'/SR_N/',sep=''))
SRN_Plot(tree_grid_search_path,paste(plot_path,'/SR_N/',sep=''))


# This part requires running all 36 combinations with rolling fitting or difference shrinkage (lambda0, lambda2 settings),
# and collect results under one csv file for plotting. 
source('4_Plots/Figure13_C6ab_Rolling_Shrinkage_SR.R')
linenames = c("Triple Sort (64)","AP-Tree (40)","Triple Sort (64) Rolling","AP-Tree (40) Rolling")
lineids = c(2,4,6,7)
ordercolid = 16
# SR_Plot_Rolling_Shrinkage(SR_Summary_File, plot_path, plotfilename, ordercolid,linenames,lineids)
# SR_Plot_Rolling_Shrinkage(SR_Summary_File, plot_path, "/RollingComparison.png", ordercolid,linenames,lineids)
linenames = c("No shrinkage","Mean shrinkage","Ridge shrinkage","Mean and ridge shrinkage")
lineids = c(1,2,3,4)
ordercolid = 5
# SR_Plot_Rolling_Shrinkage(SR_Summary_File, plot_path, "/ShrinkageTree.png", ordercolid,linenames,lineids)


# This part requires collecting all NPorts results under the same csv file for plotting
source('4_Plots/Figure14_FF10_SDF_SR_Nport.R')
# SRN_Plot(SRFF10Path,plot_path)
# SRN_Plot(SRFF10Path,plot_path)

# This part requires collecting all NPorts results under the same csv file for plotting
source('4_Plots/Figure15_FF10_SDF_SR_Nport_ML.R')
# SRN_ML_Plot(SRFF10Path,plot_path)
# SRN_ML_Plot(SRFF10Path,plot_path)

# Base portfolio Alpha test plots
source('4_Plots/FigureC8ab_XSR2.R')
# XSR2(feats_list, feat1, feat2, factor_path, port_path, port_name, plot_path, port_type)
XSR2(feats_list, feat1, feat2, factor_path, tree_grid_search_path, '/Selected_Ports_10.csv',plot_path,"tree")
XSR2(feats_list, feat1, feat2, factor_path, ts32_path,'/excess_ports.csv',plot_path,"ts32")
XSR2(feats_list, feat1, feat2, factor_path, ts64_path,'/excess_ports.csv',plot_path,"ts64")