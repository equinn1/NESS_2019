data {
  int<lower=1> I;               // # items
  int<lower=1> J;               // # persons
  int<lower=1> N;               // # observations
  int<lower=1> K;               // # years
  int<lower=1, upper=I> ii[N];  // item for n
  int<lower=1, upper=J> jj[N];  // person for n
  int<lower=1, upper=K> kk[N];  // year for n
  int<lower=0, upper=1> y[N];   // correctness for n
}
parameters {
  vector[2] xi[I];              // alpha/beta pair vectors
  vector[2] mu;                 // vector for alpha/beta means
  vector<lower=0>[2] tau;       // vector for alpha/beta residual sds
  cholesky_factor_corr[2] L_ab_Rho;
  
  vector[K] zi[J];                      // theta vectors
  vector[K] mu_theta;                   // vector for theta means
  vector<lower=0>[K] sigma_theta;       // vector for theta variances
  cholesky_factor_corr[K] L_theta_Rho;  
}
transformed parameters {
  vector[I] alpha;
  vector[I] beta;
  vector[J] thetas[K];
  vector[N] theta;
  for (i in 1:I) {
    alpha[i] = exp(xi[i,1]);
    beta[i] = xi[i,2];
  }
  for (j in 1:J) {
    for (k in 1:K) {
      for (i in 1:I) {
        theta[((j-1)*I+((k-1)*I)/2) + i] = thetas[k,j];
      }
    }
  }
}
model {
  matrix[2,2] L_ab_Sigma;
  matrix[K,K] L_theta_Sigma;
  
  L_ab_Sigma = diag_pre_multiply(tau, L_ab_Rho); // covariance matrix of a,b
  for (i in 1:I)
    xi[i] ~ multi_normal_cholesky(mu, L_ab_Sigma);
  L_ab_Rho ~ lkj_corr_cholesky(4);
  tau ~ exponential(.1);
  
  L_theta_Sigma = diag_pre_multiply(sigma_theta, L_theta_Rho); // covariance matrix of theta
  for (j in 1:J)
    zi[j] ~ multi_normal_cholesky(mu, L_theta_Sigma);
  L_theta_Rho ~ lkj_corr_cholesky(4);
  sigma_theta ~ exponential(.1);
  
  y ~ bernoulli_logit(alpha[ii] .* (theta - beta[ii]));
}
generated quantities {
  corr_matrix[2] Omega;
  Omega = multiply_lower_tri_self_transpose(L_ab_Rho);
}
