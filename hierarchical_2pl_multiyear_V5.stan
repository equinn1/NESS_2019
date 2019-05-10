data {
  int<lower=1> I;               // # items
  int<lower=1> J;               // # persons
  int<lower=1> N;               // # observations
  int<lower=1> K;               // # years
  int<lower=0> LKJF1;            // #parameter for LKJ distribution 
  int<lower=0> LKJF2;            // #parameter for LKJ distribution 
  int<lower=1, upper=I> ii[N];  // item for n
  int<lower=1, upper=J> jj[N];  // person for n
  int<lower=1, upper=K> kk[N];  // year for n
  int<lower=0, upper=1> y[N];   // correctness for n
}
transformed data{
  vector[K] ones;
  vector[K] zeros;
  for (k in 1:K){
    ones[k] = 1.0;
    zeros[k] = 0.0;
  }
}
parameters {
  vector[2] xi[I,K];            // alpha/beta pair vectors
  vector[2] mu;                 // vector for alpha/beta means
  vector<lower=0>[2] tau;       // vector for alpha/beta residual sds
  cholesky_factor_corr[2] L_ab_Rho;
  
  vector[K] zi[J];                    // theta vectors
  vector[K] mu_theta;                   // vector for theta means
  vector<lower=0>[K] sigma_theta;       // vector for theta variances
  cholesky_factor_corr[K] L_theta_Rho;  
}
transformed parameters {
  vector[N] xtheta;
  vector[N] xbeta;
  vector[N] xalpha;
  {
    int ix=0;

    for (j in 1:J) {
      for (i in 1:I) {
        for (k in 1:K) {
          ix+=1;
          xalpha[ix] = exp(xi[i,k,1]);
          xbeta[ix]  = xi[i,k,2];
        }
      }
    }
  }
  {
    int ix=0;
    for (j in 1:J) {
      for (k in 1:K) {
        for (i in 1:I) {
          ix += 1;
          xtheta[ix] = zi[j,k];
        }
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
  L_ab_Rho ~ lkj_corr_cholesky(LKJF1);
  tau ~ cauchy(0,1);
  mu ~ normal(0,1);
  
  sigma_theta ~ cauchy(0,1);
  mu_theta ~ normal(0,1);
  
  L_theta_Sigma = diag_pre_multiply(sigma_theta, L_theta_Rho);    
  for (j in 1:J) 
    zi[j] ~ multi_normal_cholesky(mu_theta, L_theta_Sigma);  
  L_theta_Rho ~ lkj_corr_cholesky(LKJF2);

  y ~ bernoulli_logit(xalpha .* (xtheta - xbeta));
}
generated quantities {
  corr_matrix[2] Omega;
  Omega = multiply_lower_tri_self_transpose(L_ab_Rho);
}
