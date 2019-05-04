# Load R packages
library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
library(ggplot2)

# Set paramters for the simulated data
I <- 34  #<- 20
J <- 200 #<- 1000
mu <- c(0, 0)
tau <- c(0.25, 1)
Omega <- matrix(c(1, 0.3, 0.3, 1), ncol = 2)

# Calculate or sample remaining paramters
Sigma <- tau %*% t(tau) * Omega
xi <- MASS::mvrnorm(I, c(0, 0), Sigma)
alpha <- exp(mu[1] + as.vector(xi[, 1]))
beta <- as.vector(mu[2] + xi[, 2])
theta <- rnorm(J, mean = 0, sd = 1)

# Assemble data and simulate response
data_list <- list(I = I, J = J, N = I * J, ii = rep(1:I, times = J), jj = rep(1:J, 
                                                                              each = I))
eta <- alpha[data_list$ii] * (theta[data_list$jj] - beta[data_list$ii])
data_list$y <- as.numeric(boot::inv.logit(eta) > runif(data_list$N))

# Fit model to simulated data
sim_fit <- stan(file = "hierarchical_2pl.stan", data = data_list, chains = 4, 
                iter = 4000)

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

# Use data and scoring function from the mirt package
library(mirt)
sat <- key2binary(SAT12, key = c(1, 4, 5, 2, 3, 1, 2, 1, 3, 1, 2, 4, 2, 1, 5, 
                                 3, 4, 4, 1, 4, 3, 3, 4, 1, 3, 5, 1, 3, 1, 5, 4, 5))

# Assemble data list and fit model
sat_list <- list(I = ncol(sat), J = nrow(sat), N = length(sat), ii = rep(1:ncol(sat), 
                                                                         each = nrow(sat)), jj = rep(1:nrow(sat), times = ncol(sat)), y = as.vector(sat))
sat_fit <- stan(file = "hierarchical_2pl.stan", data = sat_list, chains = 4, 
                iter = 500)

ex_summary <- as.data.frame(summary(sat_fit)[[1]])
ex_summary$Parameter <- as.factor(gsub("\\[.*]", "", rownames(ex_summary)))
ggplot(ex_summary) + aes(x = Parameter, y = Rhat, color = Parameter) + geom_jitter(height = 0, 
                              width = 0.5, show.legend = FALSE) + ylab(expression(hat(italic(R))))

# View table of parameter posteriors
print(sat_fit, pars = c("alpha", "beta", "mu", "tau", "Omega[1,2]"))

# Assesmble a data frame of item parameter estimates and pass to ggplot
ab_df <- data.frame(Discrimination = ex_summary[paste0("alpha[", 1:sat_list$I, 
                    "]"), "mean"], Difficulty = ex_summary[paste0("beta[", 1:sat_list$I, "]"), 
                    "mean"], parameterization = "alpha & beta")
xi_df <- data.frame(Discrimination = ex_summary[paste0("xi[", 1:sat_list$I, 
                    ",1]"), "mean"], Difficulty = ex_summary[paste0("xi[", 1:sat_list$I, ",2]"), 
                    "mean"], parameterization = "xi")
full_df <- rbind(ab_df, xi_df)
ggplot(full_df) + aes(x = Difficulty, y = Discrimination) + geom_point() + facet_wrap(~parameterization, 
                                                                  scales = "free")

