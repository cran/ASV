\name{ReportMCMC}
\alias{ReportMCMC}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Summary statistics, diagnostic statistics and plots.
%%  ~~function to do ... ~~
}
\description{
This function reports summary statistics of the MCMC samples such as the posterior 
mean, the posterior standard deviation, the 95\% credible interval, the expected sample size, the inefficiency factor, the posterior probability that the parameter is positive. Further it plots the sample path, the sample autocorrelation function
and the estimated posterior density.
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
%ReportMCMC(mx, dBm = NULL, vname = NULL, soutfilename = NULL)
ReportMCMC(mx, dBm = NULL, vname = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{mx}{nSim x m matrix where nSim is the MCMC sample size and m is the 
  number of parameters.
%%     ~~Describe \code{mx} here~~
}
  \item{dBm}{The bandwidth to compute the inefficient factor. Default value is the 
  maximum integer less than or equal to 2*sqrt(nSim)+1.
%%     ~~Describe \code{dBm} here~~
}
  \item{vname}{The vector of variable names. Default names are Param1, Param2 and so forth.
%%     ~~Describe \code{vname} here~~
}
%  \item{soutfilename}{The prefix of the output file name. Default is "Out".
%%     ~~Describe \code{soutfilename} here~~
%}
}
%\details{
%%  ~~ If necessary, more details than the description above ~~
%}
\value{
\item{Mean}{The posterior mean of the parameter}
\item{Std Dev}{The posterior standard deviation of the parameter}
\item{95\%L}{The lower limit of the 95\% credible interval of the parameter}
\item{Median}{The posterior median of the parameter}
\item{95\%U}{The upper limit of the 95\% credible interval of the parameter}
\item{ESS}{Expected sample size defined as the MCMC sample size divided by IF }
\item{IF}{Inefficiency factor. See, for example, Kim, Shephard and Chib (1998).}
\item{CD}{p-value of convergence diagnostics test by Geweke (1992). H_0:mean of the first 10\% of MCMC samples is equal to mean of the last 50\% of MCMC samples vs. H_1:not H_0.}
\item{Pr(+)}{The posterior probability that the parameter is positive.}
Further, it plots the sample path, the sample autocorrelation function 
and the posterior density for each parameter.
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}
\references{
%% ~put references to the literature/web site here ~
Kim, S., Shephard, N. and S. Chib (1998) "Stochastic volatility: likelihood inference and comparison with ARCH models", The Review of Economic Studies, 65(3), 361-393.

Geweke, J. (1992), "Evaluating the accuracy of sampling-based approaches to calculating posterior moments,"" in Bayesian Statistics 4 (ed J.M. Bernardo, J.O. Berger, A.P. Dawid and A.F.M. Smith), Oxford, UK.
}
\author{
Yasuhiro Omori
}
\note{
`freqdom' package needs to be pre-installed.
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

%\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
%}
\examples{
nobs = 80; # n is often larger than 1000 in practice.
mu = 0; phi = 0.97; sigma_eta = 0.3; rho = 0.0;
h  = 0;   Y = c();

for(i in 1:nobs){
  eps = rnorm(1, 0, 1)
  eta = rho*sigma_eta*eps + sigma_eta*sqrt(1-rho^2)*rnorm(1, 0, 1)
  y   = eps * exp(0.5*h)
  h   = mu + phi * (h-mu) + eta
  Y   = append(Y, y)
}

# This is a toy example. Increase nsim and nburn
# until the convergence of MCMC in practice.

nsim = 500; nburn = 100;
vhyper = c(0.0,1000,1.0,1.0,0.01,0.01)
out  = sv_mcmc(Y, nsim, nburn, vhyper)
vmu = out[[1]]; vphi = out[[2]]; vsigma_eta = out[[3]]; mh  = out[[4]];
myname = c(expression(mu), expression(phi),expression(sigma[eta]))
ReportMCMC(cbind(vmu,vphi,vsigma_eta), vname=myname)
%expression(phi),expression(sigma[eta])), soutfilename=c("SV"))
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory (show via RShowDoc("KEYWORDS")):
% \keyword{ ~kwd1 }
% \keyword{ ~kwd2 }
% Use only one keyword per line.
% For non-standard keywords, use \concept instead of \keyword:
% \concept{ ~cpt1 }
% \concept{ ~cpt2 }
% Use only one concept per line.
