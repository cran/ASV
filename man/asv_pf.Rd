\name{asv_pf}
\alias{asv_pf}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Particle filter for stochastic volatility models with leverage
%%  ~~function to do ... ~~
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
The function computes the log likelihood given (mu, phi, sigma_eta, rho) for stochastic volatility models with leverage  (asymmetric stochastic volatility models).
}
\usage{
asv_pf(mu, phi, sigma_eta, rho, Y, I)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{mu}{parameter value such as the posterior mean of mu
}
  \item{phi}{parameter value such as the posterior mean of phi
}
  \item{sigma_eta}{parameter value such as the posterior mean of sigma_eta
}
  \item{rho}{parameter value such as the posterior mean of rho
}
  \item{Y}{T x 1 vector (y(1),...,y(T))' of returns where T is a sample size.
}
  \item{I}{Number of particles to approximate the filtering density.
}
}
%\details{
%%  ~~ If necessary, more details than the description above ~~
%}
\value{Logarithm of the likelihood of Y given parameters (mu, phi, sigma_eta, rho)
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}
\references{
%% ~put references to the literature/web site here ~
Omori, Y., Chib, S., Shephard, N., and J. Nakajima (2007), "Stochastic volatility model with leverage: fast and efficient likelihood inference," Journal of Econometrics, 140-2, 425-449.

Takahashi, M., Omori, Y. and T. Watanabe (2022+), Stochastic volatility and  realized stochastic volatility models. JSS Research Series in Statistics, in press. Springer, Singapore.
}
\author{Yasuhiro Omori, Ryuji Hashimoto}
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
mu = 0; phi = 0.97; sigma_eta = 0.3; rho = -0.3;
h  = 0;   Y = c();
for(i in 1:nobs){
  eps = rnorm(1, 0, 1)
  eta = rho*sigma_eta*eps + sigma_eta*sqrt(1-rho^2)*rnorm(1, 0, 1)
  y   = eps * exp(0.5*h)
  h   = mu + phi * (h-mu) + eta
  Y   = append(Y, y)
}
npart = 5000 
asv_pf(mu, phi, sigma_eta, rho, Y, npart)
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
