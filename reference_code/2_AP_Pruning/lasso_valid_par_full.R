# setwd(substring(dirname(rstudioapi::getSourceEditorContext()$path), 1, nchar(dirname(rstudioapi::getSourceEditorContext()$path)) - 6))
# library(tcltk2)
# library(RSpectra)

lasso_valid_full <- function(ports, lambda0, lambda2, main_dir, sub_dir, adj_w, n_train_valid = 360, cvN = 3, runFullCV=FALSE,kmin=5, kmax=50, RunParallel,ParallelN = 10) {
  dir.create(file.path(main_dir, sub_dir), showWarnings = FALSE)
  
  ports_test = ports[(n_train_valid + 1):(nrow(ports)),]
  
  n_valid = n_train_valid / cvN
  n_train = n_train_valid - n_valid
  
  if (runFullCV){
    for (i in 1:cvN){
      ports_train = ports[-c(((i - 1) * n_valid + 1):(i * n_valid), (n_train_valid + 1):(nrow(ports))),]
      ports_valid = ports[((i - 1) * n_valid + 1):(i * n_valid),]
      lasso_cv_helper(ports_train, ports_valid, ports_test, lambda0, lambda2, main_dir, sub_dir, adj_w, nrow(ports), paste('cv', i, sep='_'), kmin,kmax,RunParallel,ParallelN)
    }
  }else{
    for (i in cvN:cvN){
      ports_train = ports[-c(((i - 1) * n_valid + 1):(i * n_valid), (n_train_valid + 1):(nrow(ports))),]
      ports_valid = ports[((i - 1) * n_valid + 1):(i * n_valid),]
      lasso_cv_helper(ports_train, ports_valid, ports_test, lambda0, lambda2, main_dir, sub_dir, adj_w, nrow(ports), paste('cv', i, sep='_'), kmin,kmax,RunParallel,ParallelN)
    }
  }
  
  
  # After pin-down the parameter, do another fit on the whole train+valid time period
  ports_train = ports[1:n_train_valid,]
  lasso_cv_helper(ports_train, NULL, ports_test, lambda0, lambda2, main_dir, sub_dir, adj_w, nrow(ports), 'full', kmin,kmax,RunParallel,ParallelN)
}

lasso_cv_helper <- function(ports_train, ports_valid, ports_test, lambda0, lambda2, main_dir, sub_dir, adj_w, n_total, cv_name, kmin=5, kmax = 50, RunParallel,ParallelN) {
  # Converting the optimization into a regression problem
  mu = apply(ports_train, 2, mean)
  sigma = cov(ports_train)
  
  mu_bar = mean(mu)
  gamma = min(dim(ports_train))
  decomp = eigen(sigma)
  gamma = min(gamma, sum(decomp$values > 1e-10))
  D = decomp$values[1:gamma]
  V = decomp$vectors[,1:gamma]
  
  sigma_tilde = V %*% diag(sqrt(D)) %*% t(V)
  mu_tilde = V %*% diag(1/sqrt(D)) %*% t(V) %*% 
    (matrix(rep(mu, length(lambda0)), nrow = length(mu), ncol = length(lambda0)) + 
       matrix(rep(lambda0, each = length(mu)) * mu_bar, nrow = length(mu), ncol = length(lambda0)))
  
  w_tilde = V %*% diag(1/D) %*% t(V) %*% 
    (matrix(rep(mu, length(lambda0)), nrow = length(mu), ncol = length(lambda0)) + 
       matrix(rep(lambda0, each = length(mu)) * mu_bar, nrow = length(mu), ncol = length(lambda0)))
  
  # Perform EN regression with CPU parallel computing
  source_path = '2_AP_Pruning/lasso.R'
  if (RunParallel){
    library(doSNOW)
    library(foreach)
    cl<-makeCluster(ParallelN)
    registerDoSNOW(cl)
    foreach(i=c(1:length(lambda0))) %dopar% {
      source(source_path, local = TRUE)
      
      for (j in 1: length(lambda2)){
        lasso_results = lasso(sigma_tilde, mu_tilde[,i], lambda2[j], 100, kmin,kmax)
        
        train_SR = numeric(nrow(lasso_results[[1]]))
        if (!is.null(ports_valid)) {
          valid_SR = numeric(nrow(lasso_results[[1]]))
        }
        test_SR = numeric(nrow(lasso_results[[1]]))
        betas = matrix(NA, nrow = nrow(lasso_results[[1]]), ncol = length(mu))
        portsN = lasso_results[[2]]
        
        for (r in 1:nrow(lasso_results[[1]])){
          b = lasso_results[[1]][r,]
          b = b * adj_w
          b = b / abs(sum(b))
          
          sdf_train = as.matrix(ports_train) %*% (b / adj_w)
          if (!is.null(ports_valid)) {
            sdf_valid = as.matrix(ports_valid) %*% (b / adj_w)
          }
          
          sdf_test = as.matrix(ports_test) %*% (b / adj_w)
          
          train_SR[r] = mean(sdf_train)/sd(sdf_train)
          if (!is.null(ports_valid)) {
            valid_SR[r] = mean(sdf_valid)/sd(sdf_valid)
          }
          test_SR[r] = mean(sdf_test)/sd(sdf_test)
          betas[r,] = b
        }
        
        colnames(betas) = colnames(ports_train)
        if (!is.null(ports_valid)) {
          results = cbind(train_SR, valid_SR, test_SR, portsN, betas)
        } else{
          results = cbind(train_SR, test_SR, portsN, betas)
        }
        
        write.table(results, paste(file.path(main_dir, sub_dir), '/results_',cv_name,'_l0_', i,'_l2_',j,'.csv', sep=''), sep=',',row.names=F)
      }
    }
    stopCluster(cl)
  }else{
    for (i in c(1:length(lambda0))){
      source(source_path, local = TRUE)
      
      for (j in 1: length(lambda2)){
        lasso_results = lasso(sigma_tilde, mu_tilde[,i], lambda2[j], 100, kmin,kmax)
        
        train_SR = numeric(nrow(lasso_results[[1]]))
        if (!is.null(ports_valid)) {
          valid_SR = numeric(nrow(lasso_results[[1]]))
        }
        test_SR = numeric(nrow(lasso_results[[1]]))
        betas = matrix(NA, nrow = nrow(lasso_results[[1]]), ncol = length(mu))
        portsN = lasso_results[[2]]
        
        for (r in 1:nrow(lasso_results[[1]])){
          b = lasso_results[[1]][r,]
          b = b * adj_w
          b = b / abs(sum(b))
          
          sdf_train = as.matrix(ports_train) %*% (b / adj_w)
          if (!is.null(ports_valid)) {
            sdf_valid = as.matrix(ports_valid) %*% (b / adj_w)
          }
          
          sdf_test = as.matrix(ports_test) %*% (b / adj_w)
          
          train_SR[r] = mean(sdf_train)/sd(sdf_train)
          if (!is.null(ports_valid)) {
            valid_SR[r] = mean(sdf_valid)/sd(sdf_valid)
          }
          test_SR[r] = mean(sdf_test)/sd(sdf_test)
          betas[r,] = b
        }
        
        colnames(betas) = colnames(ports_train)
        if (!is.null(ports_valid)) {
          results = cbind(train_SR, valid_SR, test_SR, portsN, betas)
        } else{
          results = cbind(train_SR, test_SR, portsN, betas)
        }
        
        write.table(results, paste(file.path(main_dir, sub_dir), '/results_',cv_name,'_l0_', i,'_l2_',j,'.csv', sep=''), sep=',',row.names=F)
      }
    }
  }
}