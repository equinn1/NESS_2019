library(rstan)
library(mvtnorm)

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

N = 500
K=4

s = matrix(c(2.0,1.5,1.2,1.0,  
             1.5,3.0,1.0,1.0,  
             1.2,1.0,3.5,1.0,  
             1.0,1.0,1.0,3.0),nrow=4)
s

y = rmvnorm(N,mean=c(0.2,0.3,0.4,0.5),sigma=s)
#y

var(y)
cor(y)
mean(y[,1])
mean(y[,2])

stanfit = stan("mvnormal_test.stan")

summary(stanfit)

pd = extract(stanfit)

mean(pd$Omega[,1,2])




