library(lars)

# Call LARS to calculate the whole path for EN regularized regression
lasso <- function(X, y, lambda2, steps = 70, kmin = 5, kmax = 50) {
  n = nrow(X)
  p = ncol(X)
  yy = c(y, rep(0, p))
  XX = rbind(X, diag(sqrt(lambda2), p, p))
  
  lasso_obj = lars(XX, yy, type="lasso", normalize = FALSE, intercept = FALSE)
  beta = coef(lasso_obj)
  
  K = apply(beta, 1, function(x){return(sum(x != 0))})
  subset = K >= kmin & K <= kmax
  return(list(beta[subset,], K[subset]))
}