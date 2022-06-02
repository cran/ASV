\name{ASV-package}
\alias{ASV-package}
\alias{ASV}
\docType{package}
\title{
\packageTitle{ASV}
}
\description{
This function estimates model parameters and latent log volatilities for 
stochastic volatility models:  

y(t) = eps(t)*exp(h(t)/2), h(t+1) = mu + phi*(h(t)-mu) + eta(t)

eps(t)~i.i.d. N(0,1), eta(t)~i.i.d. N(0,sigma_eta^2)

where we assume the correlation between eps(t) and eta(t) equals to rho.
}
\details{
%The DESCRIPTION file:
%\packageDESCRIPTION{ASV}
%\packageIndices{ASV}
The highly efficient Markov chain Monte Carlo algorithm is based on the mixture 
sampler by Omori, Chib, Shephard and Nakajima (2007), but it further corrects the approximation error within the sampling algorithm.
See Takahashi, Omori and Watanabe (2022+) for more details.}

%\author{Yasuhiro Omori, Ryuji Hashimoto}
\references{
%~~ Literature or other references for background information ~~
Omori, Y., Chib, S., Shephard, N., and J. Nakajima (2007), "Stochastic volatility model with leverage: fast and efficient likelihood inference," Journal of Econometrics, 140-2, 425-449.

Takahashi, M., Omori, Y. and T. Watanabe (2022+), Stochastic volatility and  realized stochastic volatility models. JSS Research Series in Statistics, in press. Springer, Singapore.
}
%~~ Optionally other standard keywords, one ~~
%~~ per line, from file KEYWORDS in the R ~~
%~~ documentation directory ~~
\keyword{Markov chain Monte Carlo Stochastic volatility Leverage Asymmetry}
\seealso{
%~~ Optional links to other man pages, e.g. ~~
%~~ \code{\link[<pkg>:<pkg>-package]{<pkg>}} ~~
\code{\link{sv_mcmc}, \link{asv_mcmc}, \link{sv_pf}, \link{asv_pf}, \link{sv_apf}, \link{asv_apf}}
}
%\examples{
%~~ simple examples of the most important functions ~~
%}
%\note{The author thanks Ryuji Hashimoto for his precious assistance.}