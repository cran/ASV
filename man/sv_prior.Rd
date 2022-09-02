\name{sv_prior}
\alias{sv_prior}
\title{
Compute the logarithm of the prior density for the stochastic volatility models without leverage
}
\description{
This function computes the logarithm of the prior density for 
stochastic volatility models without leverage (symmetric stochastic volatility models):  

mu~N(mu_0,sigma_0^2), (phi+1)/2~Beta(a_0,b_0), sigma_eta^2~IG(n_0/2,S_0/2)
}
\usage{
sv_prior(Theta_star, vHyper = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{Theta_star}{
   a vector of parameters to evaluate the prior density: Theta_star = c(mu, phi, sigma_eta)
}
  \item{vHyper}{
   a vector of hyper-parameters to evaluate the prior density: vHyper     = c(mu_0, sigma_0, a_0, b_0, n_0, S_0)
}
}
%\details{
%%  ~~ If necessary, more details than the description above ~~
%}
\value{
The logarithm of the prior density.
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}
%\references{
%% ~put references to the literature/web site here ~
%}
\author{Yasuhiro Omori}
%\note{
%%  ~~further notes~~
%}

%% ~Make other sections like Warning with \section{Warning }{....} ~

%\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
%}
\examples{
vhyper     = c(0, 1, 20, 1.5, 5, 0.05)
theta_star = c(0, 0.97, 0.3) 
sv_prior(theta_star, vhyper)
}

