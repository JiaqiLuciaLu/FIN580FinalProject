
Validating sample. The middle 10 years (1984 – 1993, 120 monthly observations) of returns
serve as the validation set for hyperparameter tuning: We pick the model based on Sharpe ratio of
the tangency portfolio on the validation dataset, fixing the SDF weights at their training values.
Table ?? reports the hyperparameters we used for the SDF construction. For each combination
of (λ0,K,λ2) the lasso penalty λ1 is chosen such that the number of non-zero weights reaches the
target number K. In particular, we tune the value of λ1 for AP-Trees to select 40 portfolios, which
makes the dimension of pruned trees comparable to Fama-French triple-sorted 32 and 64 portfolios.
Testing sample. The last 23 years of monthly data are used to compare basis assets, recovered
by AP-Trees and triple-sorting. We fix portfolio weights and their selection at the values, estimated
on the training sample, and tuning parameters chosen with the validation, making, therefore, all the
performance metrics effectively out-of-sample. We focus on Sharpe ratios and