\name{sv_logML}
\alias{sv_logML}
\title{
Compute the logarithm of the marginal likelihood for the stochastic volatility models without leverage 
}

\description{
This function computes the logarithm of the marginal likelihood for stochastic volatility models without leverage (symmetric stochastic volatility models): }

\usage{
sv_logML(H, Theta, Theta_star, Y, iI = NULL, iM = NULL, vHyper = NULL)
}

\arguments{
  \item{H}{
    T x 1 vector of latent log volatilities to start the reduced MCMC run to compute the log posterior density.}
  \item{Theta}{
   a vector of parameters to start the reduced MCMC run to compute the log posterior density. Theta = c(mu, phi, sigma_eta)}
  \item{Theta_star}{
   a vector of parameters to evaluate the log posterior density. Theta_star = c(mu, phi, sigma_eta)}
  \item{Y}{
   T x 1 vector of returns}
  \item{iI}{the number of particles to approximate the filtering density. Default is 5000.}
  \item{iM}{
   the number of iterations for the reduced MCMC run. Default is 5000.
}
  \item{vHyper}{
   a vector of hyper-parameters to evaluate the log posterior density. vHyper     = c(mu_0, sigma_0, a_0, b_0, n_0, S_0). Defaults is (0,1000, 1, 1, 0.01, 0.01)
}

}
%\details{
%%  ~~ If necessary, more details than the description above ~~
%}
\value{
4 x 2 matrix. 
  \item{Column 1}{The logarithms of the marginal likelihood, the likelihood, the prior density and the posterior density.}
  \item{Column 2}{The standard errors of the logarithms of the marginal likelihood, the likelihood, the prior density and the posterior density.}
}
\references{
Chib, S., and Jeliazkov, I. (2001). Marginal likelihood from the Metropolis-Hastings output. Journal of the American statistical association, 96(453), 270-281.
}
\author{
Yasuhiro Omori
}
%\note{
%%  ~~further notes~~
%}

%% ~Make other sections like Warning with \section{Warning }{....} ~

%\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
%}
\examples{
set.seed(111)
nobs = 80; # n is often larger than 1000 in practice.
mu = 0; phi = 0.97; sigma_eta = 0.3;
h  = 0;   Y = c();
for(i in 1:nobs){
  eps = rnorm(1, 0, 1)
  eta = sigma_eta*rnorm(1, 0, 1)
  y   = eps * exp(0.5*h)
  h   = mu + phi * (h-mu) + eta
  Y   = append(Y, y)
}

# This is a toy example. Increase nsim and nburn
# until the convergence of MCMC in practice.
nsim = 300; nburn = 100;
vhyper = c(0.0,1000,1.0,1.0,1.0,1.0,0.01,0.01)
out  = sv_mcmc(Y, nsim, nburn, vhyper)
vmu = out[[1]]; vphi = out[[2]]; vsigma_eta = out[[3]]; mh  = out[[4]];
mu  = mean(vmu); phi = mean(vphi); sigma_eta = mean(vsigma_eta);
#
h          = mh[nsim,]
theta      = c(vmu[nsim],vphi[nsim],vsigma_eta[nsim])
theta_star = c(mu, phi, sigma_eta)

# Increase iM in practice (such as iI = 5000, iM =5000).
result = sv_logML(h, theta, theta_star, Y, 100, 100, vHyper = vhyper)
result1     = matrix(0, 4, 2)
result1[,1] =result[[1]]
result1[,2] =result[[2]]
  
colnames(result1) = c("Estimate", "Std Err")
rownames(result1) = c("Log marginal lik", "Log likelihood", "Log prior", "Log posterior")
print(result1, digit=4)
}