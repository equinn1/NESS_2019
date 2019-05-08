# Load R packages
#rm(list=ls())
library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
library(ggplot2)
library(mvtnorm)

set.seed(12345)

# Set paramters for the simulated data
I <- 34  # questions
J <- 100 # students
K <- 2   # years
N <- I*J*K  # item responses

#means and variance-covariance matrix for IRT parameters
mu <- rep(c(0, 0),K)
tau <- rep(c(0.25, 1),K)
variance_block <- matrix(c(1, 0.3, 0.3, 1), ncol = 2)
Omega <- matrix(rep(0,4*K^2),nrow=K*2)

for (k in 1:K){
  Omega[(2*k-1):(2*k),(2*k-1):(2*k)] <- variance_block
}

# Calculate or sample remaining paramters
Sigma <- tau %*% t(tau) * Omega
xi <- MASS::mvrnorm(I, rep(0,2*K), Sigma)
alpha=numeric()
beta= numeric()
for (i in seq(1,K)){
  alpha=c(alpha,exp(mu[(2*i-1)] + as.vector(xi[, (2*i-1)])))
  beta =c(beta,beta <- as.vector(mu[(2*i)] + xi[, (2*i)]))
}

theta_rho = -1/2 + runif(J)

theta_m = matrix(rep(0,J*K),nrow=J)
theta_m
for (i in seq(1:J)){
  theta_sigma = matrix(c(1,theta_rho[i],theta_rho[i],1),nrow=2)
  theta_m[i,] = rmvnorm(1,mean=rep(0,K),sigma=theta_sigma)
}

# Assemble data and simulate response
ii = rep(rep(1:I,times=K), times=J)     #column of item numbers 
jj = rep(1:J,each=(K*I))                #column of student numbers
kk = rep(rep(1:K,each=I),times=J)       #column of year numbers

eta=rep(0,N)                            # a(theta-b)
y = rep(0,N)                            # question responses

for (i in 1:N){
  eta[i] <- alpha[ii[i]] * (theta_m[jj[i],kk[i]] - beta[ii[i]])
  y[i] <- as.numeric(boot::inv.logit(eta[i]) > runif(1))  
}

LKJF = 5

data_list <- list(I=I,J=J, K=K, N=N, y=y, LKJF=LKJF)

# Fit model to simulated data
sim_fit <- stan(file = "hierarchical_2pl_multiyear_V3.stan", data=data_list, chains = 4, 
                iter = 4000)

summary(sim_fit)

pd <- extract(sim_fit)

k=22
mean(pd$zi[,k,1])
mean(pd$zi[,k,2])
cov(pd$zi[,k,])

library(shinystan)

launch_shinystan(sim_fit)


sim_summary <- as.data.frame(summary(sim_fit)[[1]])
sim_summary$Parameter <- as.factor(gsub("\\[.*]", "", rownames(sim_summary)))
ggplot(sim_summary) + aes(x = Parameter, y = Rhat, color = Parameter) + geom_jitter(height = 0, 
               width = 0.5, show.legend = FALSE) + ylab(expression(hat(italic(R))))

# Make vector of wanted parameter names
wanted_pars <- c(paste0("alpha[", 1:I, "]"), paste0("beta[", 1:I, "]"), c("mu[1]", 
                                                                          "mu[2]", "tau[1]", "tau[2]", "Omega[1,2]"))

# Get estimated and generating values for wanted parameters
generating_values = c(alpha, beta, mu, tau, Omega[1, 2])
estimated_values <- sim_summary[wanted_pars, c("mean", "2.5%", "97.5%")]

# Assesmble a data frame to pass to ggplot()
sim_df <- data.frame(parameter = factor(wanted_pars, rev(wanted_pars)), row.names = NULL)
sim_df$middle <- estimated_values[, "mean"] - generating_values
sim_df$lower <- estimated_values[, "2.5%"] - generating_values
sim_df$upper <- estimated_values[, "97.5%"] - generating_values

# Plot the discrepancy
ggplot(sim_df) + aes(x = parameter, y = middle, ymin = lower, ymax = upper) + 
  scale_x_discrete() + geom_abline(intercept = 0, slope = 0, color = "white") + 
  geom_linerange() + geom_point(size = 2) + labs(y = "Discrepancy", x = NULL) + 
  theme(panel.grid = element_blank()) + coord_flip()
