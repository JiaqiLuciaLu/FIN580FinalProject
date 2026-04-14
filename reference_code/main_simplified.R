# This code is tested under the following package versions with a 2020 Macbook Pro with M1 CPU
# This simplified version without full grid search takes about 20 mins to run with CPU parallel computing with 6 cores (pralleln = 6)
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

# Change this to FALSE for max compatibility
RunParallel = TRUE
pralleln = 6
# RunParallel = FALSE
# pralleln = 1

raw_data_path = '../Data/ret_characteristics/'
data_chunk_path = '../Data/data_chunk_files_quantile/'
tree_portfolio_path = '../Data/tree_portfolio_quantile/'
tree_grid_search_path = '../Data/TreeGridSearch/'
factor_path = '../Data/factor/'
plot_path = '../Data/plot/'
dir.create(data_chunk_path, showWarnings = FALSE)
dir.create(tree_portfolio_path)
dir.create(tree_grid_search_path)

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

#################################################################################
################  AP Pruning on Base Portfolios with Grid Search ################
#################################################################################


# In the paper we used a much larger grid search on the following parameter, but doing the full parameter search can take a long time to finish
# lambda0 = seq(0, 0.9, 0.05)
# lambda2 = 0.1^seq(5, 8, 0.25)

# For demonstration, we put the below parameters not a full grid search
lambda0 = seq(0.15, 0.15, 0.05)
lambda2 = 0.1^seq(8, 8, 0.25)

source('2_AP_Pruning/AP_Pruning.R')
# AP_Pruning(feats_list, feat1, feat2, input_path,input_file_name, output_path,n_train_valid,cvN,runFullCV,kmax, RunParallel, ParallelN,IsTree,lambda0, lambda2)
# Due to dimension of trees, the grid search on trees can take a while without parallel computing
AP_Pruning(feats_list, feat1, feat2, tree_portfolio_path,'/level_all_excess_combined_filtered.csv',tree_grid_search_path,n_train_valid,cvN,FALSE,50, RunParallel, pralleln, TRUE,lambda0, lambda2)

#################################################################################
#############  Parameter Search and Asset Pricing Regression Tests ##############
#################################################################################


# Grid Search on Lambda tuning parameters
source('3_Metrics_Collection/Pick_Best_Lambda.R')
# pickBestLambda(feats_list, feat1, feat2, ap_prune_result_path,portN,lambda0,lambda2, portfolio_path, port_name, fullCV=FALSE, writetable=TRUE)
sharperatios = pickBestLambda(feats_list, feat1, feat2, tree_grid_search_path,portN,lambda0,lambda2, tree_portfolio_path,'/level_all_excess_combined_filtered.csv')

# Time Series Regression tests wrt factors. The results are collected to form table 1 and 2
# Re-run this on the sample where we remove small cap stocks gives table 3
source('3_Metrics_Collection/SDF_TimeSeries_Regressions.R')
# SDF_regression(feats_list, feat1, feat2, factor_path, port_path, port_name,weight_name)
regressionresults = SDF_regression(feats_list, feat1, feat2, factor_path, tree_grid_search_path, '/Selected_Ports_10.csv','/Selected_Ports_Weights_10.csv')

print(paste("Training SR:", round(sharperatios[1],4), "    Validation SR:", round(sharperatios[2],4), "    Testing SR:", round(sharperatios[3],4),sep=""))
print("SDF Regression results:")
print(regressionresults)
