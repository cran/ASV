#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins("cpp11")]]
#include "asv.h"

// [[Rcpp::depends(RcppProgress)]]
#include "progress.hpp"
#include <progress_bar.hpp>


//**********************************************//
// Stochastic volatility model without leverage //
//**********************************************//


// [[Rcpp::export]]
Rcpp::List sv_mcmc(arma::vec return_vector, // Return vector
                  Rcpp::Nullable<double> nSim  = R_NilValue,
                  Rcpp::Nullable<double> nBurn = R_NilValue,
                  Rcpp::Nullable<Rcpp::NumericVector> vHyper = R_NilValue){

  /* Return vector */
  Y = return_vector; 
  /* Sample size */
  T = Y.n_elem;
  /*---Setup---*/
  double nsim, nburn;
  
  // # of MCMC iterations
  if(nSim.isNotNull()){nsim = Rcpp::as<double>(nSim);}       
  else{nsim = 5000;}
  
  if(nBurn.isNotNull()){nburn = Rcpp::as<double>(nBurn);}       
  else{nburn = 2.0 * sqrt(nsim) + 1 ;}// # of burn-in periods
  
  /*---Initial values---*/
  double phi, sigma_eta, mu;
    mu        =  log( arma::sum(Y%Y)/T );
    phi       =  0.90;
    sigma_eta =  0.30;

  arma::vec h;
  
  h         = arma::zeros(T) + mu;
  int t;
  for(t = 1; t < T; t++){
    h[t] = mu + phi*(h[t-1] -mu) + R::rnorm(0, sigma_eta); 
  }
  
  /*---Prior distributions for theta---*/
  // theta = {mu, phi, sigma_eta}
  if(vHyper.isNotNull())
    {
    Rcpp::NumericVector vhyper(vHyper);
    mu_0    = vhyper(0); // mu~N(mu_0,sigma_0^2)
    sigma_0 = vhyper(1);
    a_0     = vhyper(2);   // (phi+1)/2~Beta(a_0,b_0)
    b_0     = vhyper(3);
    n_0     = vhyper(4);    // sigma_eta^2~IG(n_0/2,S_0/2)
    S_0     = vhyper(5);
    
    }       
  else{
    mu_0    = 0.0; // mu~N(mu_0,sigma_0^2)
    sigma_0 = 1000;
    a_0     = 1.0;   // (phi+1)/2~Beta(a_0,b_0)
    b_0     = 1.0;
    n_0     = 0.01;    // sigma_eta^2~IG(n_0/2,S_0/2)
    S_0     = 0.01;
    }

  /*---Define variables---*/
  /*------Do not edit-----------------------------------*/ 
  // For the mixture approximation: 
    p       = {0.00609, 0.04775, 0.13057, 0.20674, 0.22715,
               0.18842, 0.12047, 0.05591, 0.01575, 0.00115};
    m       = {1.92677, 1.34744, 0.73504, 0.02266, -0.85173,
               -1.97278, -3.46788, -5.55246, -8.68384, -14.65};
    v2      = {0.11265, 0.17788, 0.26768, 0.40611, 0.62699,
               0.98583, 1.57469, 2.54498, 4.16591, 7.33342};
    v       = sqrt(v2); // above v is v^2
    a       = {1.01418, 1.02248, 1.03403, 1.05207, 1.08153,
               1.13114, 1.21754, 1.37454, 1.68327, 2.50097};
    b       = {0.5071, 0.51124, 0.51701, 0.52604, 0.54076,
               0.56557, 0.60877, 0.68728, 0.84163, 1.25049};
    dcst   = 1.0E-7;          
  /*----------------------------------------------------*/
  
    
  arma::vec s, theta;
  arma::vec mu_s, phi_s, sigma_eta_s;
  Rcpp::NumericVector Y_;
  Rcpp::List result;
  
  Y_     = Rcpp::wrap(Y);   // # Declare Y as NumericVector of Rcpp
  Y_star = log(Y%Y + dcst); // log(y^2+c)

  cAccH = 0; cAccTheta = 0;
  theta = {mu, phi, sigma_eta};

  mu_s        = arma::zeros(nsim);
  phi_s       = arma::zeros(nsim);
  sigma_eta_s = arma::zeros(nsim);
  
  arma::mat h_s;
  h_s = arma::zeros(nsim, T);
  
  theta = sv_sample_theta(h, theta);
  s     = sv_sample_s(h, theta);
  h     = sv_sim_smoother(s,theta);
  
  /*---Start MCMC---*/
  Rprintf("Start MCMC estimation\n");
  Progress p1(nburn+nsim, true);
  int k;
  for(k = -nburn; k < nsim; k++){
    p1.increment();
//  if(k % 500 == 0){Rprintf("iteration=%i\n", k);}
    /* Reset counter */
    if(k == 0) { cAccH = 0; cAccTheta = 0; cRep = 0;}
    /*---Sample theta---*/
    theta = sv_sample_theta(h, theta);
    /*---Sample s ---*/
    s     = sv_sample_s(h, theta);
    /*---Sample h_t ---*/
    h     = sv_sample_h(s, h, theta);

    /*---Save MCMC samples---*/
    if(0 <= k){
      mu_s[k]        = theta[0];
      phi_s[k]       = theta[1];
      sigma_eta_s[k] = theta[2];
      h_s(k, arma::span(0,T-1)) = h.t();
    }
    cRep = cRep + 1;
  }
  Rprintf("Acceptance Rate for h     = %6.2f %% \n", cAccH/cRep*100);
  Rprintf("Acceptance Rate for (mu,phi,sigma_eta) = %6.2f %% \n", cAccTheta/cRep*100);
  
  result = Rcpp::List::create(mu_s, phi_s, sigma_eta_s, h_s);
  return(result);
}


arma::vec sv_sample_s(arma::vec h, arma::vec theta){

  int t;
  arma::vec eps_star, eta;
  arma::vec result = arma::zeros(T);
  for(t = 0; t < T; t++){

    /*---Calculate posterior density---*/
    eps_star = arma::pow((Y_star[t]-h[t]-m), 2) / (2*v%v); 

    arma::vec prob = p % (1/v) % (arma::exp(- (eps_star - arma::mean(eps_star))));
    prob           = prob / arma::sum(prob);
    Rcpp::NumericVector j = {0,1,2,3,4,5,6,7,8,9};
    double s_t     = Rcpp::sample(j, 1, true, Rcpp::wrap(prob))[0];
    result[t]      = s_t;
  }

  return(result);
}


Rcpp::List sv_kalman_filter(arma::vec s, arma::vec theta){

  double mu, phi, sigma_eta;
  mu        = theta[0];
  phi       = theta[1];
  sigma_eta = theta[2];
  
  //*---Kalman filter---*/
  int t;
  double h = mu;
  double P = sigma_eta*sigma_eta / (1-phi*phi);
  double h_star = 0;
  double A_star = -1;
  arma::vec e, D, K, L, f, F;
  arma::mat J;
  e = arma::zeros(T);
  D = arma::zeros(T);
  K = arma::zeros(T);
  L = arma::zeros(T);
  J = arma::zeros(T,2);
  f = arma::zeros(T);
  F = arma::zeros(T);

  for(t = 0; t < T; t++){
    arma::vec H = {0, sigma_eta};
    arma::vec G = {v[s[t]], 0};
    arma::vec W = {0, 1};
    arma::vec beta = {1, (1-phi)*mu};
    arma::vec B = {0, 1-phi};

    e[t] = Y_star[t] - m[s[t]] - h;
    D[t] = P + arma::dot(G,G);
    K[t] = (phi*P + arma::dot(H,G)) / D[t];
    L[t] = phi - K[t];
    J(t, arma::span(0,1)) = (H - K[t]*G).t();
    f[t] = Y_star[t] - m[s[t]] - h_star;
    F[t] = - A_star;

    h      = arma::dot(W,beta) + phi*h + K[t]*e[t];
    P      = phi*P*L[t] + arma::dot(H, J(t,arma::span(0,1)).t());
    h_star = phi*h_star + K[t]*f[t];
    A_star = phi - 1 + phi*A_star + K[t]*F[t];
  }

  Rcpp::List result = Rcpp::List::create(e, D, J, L, f, F);

  return(result);
}


arma::vec sv_sim_smoother(arma::vec s, arma::vec theta){

  /*---(1) Kalman filter---*/
  int t;
  Rcpp::List kalman_list;

  double mu, phi, sigma_eta;
  mu        = theta[0];
  phi       = theta[1];
  sigma_eta = theta[2];

    
  kalman_list = sv_kalman_filter(s, theta);
  arma::vec e = kalman_list[0];
  arma::vec D = kalman_list[1];
  arma::mat J = kalman_list[2];
  arma::vec L = kalman_list[3];

  /*---(2) Simulation Smoother---*/
  double r, U, C, kappa, V;
  arma::vec eta, result;
  r = 0;
  U = 0;
  eta = arma::zeros(T);
  result = arma::ones(T) * mu;
  for(t = T-1; t > -1; t--){
    arma::mat H  = {0, sigma_eta};
    arma::mat G  = {v[s[t]], 0};
    arma::mat mJ = J(t,arma::span(0,1));
    arma::mat I  = arma::eye(2,2);

    arma::mat matC   = H * (I - G.t()*G/D[t] - mJ.t()*mJ*U) * H.t();
    C                = matC(0,0);
    kappa            = R::rnorm(0, sqrt(C));
    arma::mat matV   = H * (G.t()/D[t] + mJ.t()*U*L[t]);
    V                = matV(0,0);
    arma::mat mateta = H * (G.t()*e[t]/D[t] + mJ.t()*r) + kappa;
    eta[t]           = mateta(0,0);
    r                = e[t]/D[t] + L[t]*r - kappa*V/C;
    U                = 1/D[t] + L[t]*U*L[t] + V*V/C;
  }

  arma::mat H_0    = {0, sigma_eta/sqrt(1-phi*phi)};
  arma::mat mJ_0   = H_0;
  arma::mat I      = arma::eye(2,2);
  arma::mat matC_0 = H_0 * (I - mJ_0.t()*mJ_0*U) * H_0.t();
  double C_0       = arma::as_scalar(matC_0);
  double kappa_0   = R::rnorm(0, sqrt(C_0));
  arma::mat mateta = H_0 * mJ_0.t()*r + kappa_0;
  double eta_0     = arma::as_scalar(mateta);
  result[0]        = mu + eta_0;

  for(t = 1; t < T; t++){
    result[t] = mu*(1-phi)+phi*result[t-1]+eta[t-1];
  }
  
  return(result);
}


double sv_loglikelihood(arma::vec s, arma::vec theta){
  
  /*---(1) Kalman filter---*/
  Rcpp::List kalman_list;
  kalman_list = sv_kalman_filter(s, theta);
  arma::vec D = kalman_list[1];
  arma::mat f = kalman_list[4];
  arma::vec F = kalman_list[5];

  /*---(2) Compute Log Likelihood---*/
  double C_1, mu_1;
  C_1  = 1 / ( 1/(sigma_0*sigma_0) + arma::sum(F%(1/D)%F) );
  mu_1 = C_1 * ( mu_0/(sigma_0*sigma_0) + arma::sum(F%(1/D)%f) );

  double result;
  result = -T*log(2*arma::datum::pi)/2 - arma::sum(log(arma::abs(D)))/2 -
    log(fabs(sigma_0*sigma_0))/2 + log(fabs(C_1))/2 -
    ( arma::sum(f%(1/D)%f) + mu_0*(1/(sigma_0*sigma_0))*mu_0 - mu_1*(1/C_1)*mu_1 )/2;

  return(result);
}

double sv_loglikelihood_theta(arma::vec s, arma::vec theta){

  /*---(1) Kalman filter---*/
  Rcpp::List kalman_list;
  kalman_list = sv_kalman_filter(s, theta);
  arma::vec D = kalman_list[1];
  arma::mat f = kalman_list[4];
  arma::vec F = kalman_list[5];
  
  /*---(2) Compute Log Likelihood given theta =(mu, phi, sigma_eta)---*/
  double result, mu;
  mu = theta[0];
  
  result = -T*log(2*arma::datum::pi)/2 - arma::sum(log(arma::abs(D)))/2 
           - ( arma::sum(f%(1/D)%f) - 2 * mu* arma::sum(F%(1/D)%f) 
               + mu * mu *arma::sum(F%(1/D)%F)  )/2;
  
  return(result);
}


double sv_theta_post_max(arma::vec x, arma::vec h){
   // x={mu, phi_, sigma_eta_}
  
  double mu, phi_, sigma_eta_, phi, sigma_eta;
  mu         = x[0];
  phi_       = x[1];
  phi        = (exp(phi_)-1) / (exp(phi_)+1);
  sigma_eta_ = x[2];
  sigma_eta  = exp(sigma_eta_);

  arma::vec theta = arma::zeros(3);
  theta[0] = mu;
  theta[1] = phi;
  theta[2] = sigma_eta; // = (1/2)*log(sigma_eta^2)

  //*---Compute Log Prior density---*/
  double log_prior_mu, log_prior_phi, log_prior_sigma_eta;
  log_prior_mu        = R::dnorm(mu, mu_0, sigma_0, true);
  log_prior_phi       = R::dbeta( (phi+1)/2, a_0, b_0, true ) - log(2);  
  // (phi+1)/2~Beta(a_0,b_0)
  log_prior_sigma_eta = R::dgamma(1/(sigma_eta*sigma_eta), n_0/2, 2/S_0, true)
    - 2 * log(sigma_eta*sigma_eta); 
  // sigma_eta^2~IG(n_0/2,S_0/2)
  
  /*---Compute Log Likelihood---*/
  double hbar, lf; // lf: log likelihood

  lf = R::dnorm( h[0], mu, sigma_eta/sqrt(1-phi*phi), true);
  int t;
  for(t = 0; t < T-1; t++){
    hbar = mu*(1-phi) + phi*h[t];
    if( t < T-1 ){
      lf = lf + R::dnorm( Y[t], 0.0, exp(0.5*h[t]), true )
              + R::dnorm( h[t+1], hbar, sigma_eta, true);
    }
    else{
      lf = lf + R::dnorm( Y[t], 0.0, exp(0.5*h[t]), true );
    }
  }

  /*---Compute Jacobian---*/
  double jacobian = phi_ + log(2.0) + 2*sigma_eta_ + log(2.0)- 2*log(exp(phi_)+1);
  /*---Compute Log Posterior density---*/
  double result = lf + log_prior_phi + log_prior_sigma_eta + log_prior_mu + jacobian;
  return(result);
}


arma::vec sv_deriv1(arma::vec x, arma::vec h){
                 // x={mu, phi_, sigma_eta_}
  int i;
  arma::vec e = arma::zeros(3);
  double epsilon = 0.001;
  arma::vec grad = arma::zeros(3);

  for(i = 0; i < 3; i++){
    e[i] = 1;
    grad[i] = ( 
      sv_theta_post_max(x+epsilon*e, h) - sv_theta_post_max(x-epsilon*e, h) 
              )/ (2*epsilon);
      e[i] = 0;
  }
  
  return(grad);
}


arma::mat sv_deriv2(arma::vec x, arma::vec h){
                 // x={mu, phi, sigma_eta}
  int i, j;
  arma::vec e_i, e_j;
  e_i = arma::zeros(3);
  e_j = arma::zeros(3);
  double epsilon = 0.001;

  arma::mat hess = arma::zeros(3,3);

  for(i = 0; i < 3; i++){
    for(j = 0; j < 3; j++){
      e_i[i] = 1;
      e_j[j] = 1;
      hess(i,j) = (
          sv_theta_post_max(x+epsilon*e_i+epsilon*e_j, h)
        + sv_theta_post_max(x-epsilon*e_i-epsilon*e_j, h)
        - sv_theta_post_max(x-epsilon*e_i+epsilon*e_j, h)
        - sv_theta_post_max(x+epsilon*e_i-epsilon*e_j, h)
                    ) / (4*epsilon*epsilon);
      e_i[i] = 0;
      e_j[j] = 0;
    }
  }

  return(hess);
}


arma::vec sv_Opt(arma::vec x, arma::vec h){
  // x={mu, phi_, sigma_eta_}
              
  Rcpp::Environment stats("package:stats");
  Rcpp::Function optim = stats["optim"];

  Rcpp::List control = Rcpp::List::create(Rcpp::Named("fnscale") = -1.0);
  Rcpp::List out = optim(Rcpp::_["par"]    = x,
                         // Make sure this function is not exported!
                         Rcpp::_["fn"] = Rcpp::InternalFunction(&sv_theta_post_max),
                         Rcpp::_["h"] = h,
                         Rcpp::_["method"]  = "BFGS", 
                         Rcpp::_["control"] = control,
                         Rcpp::_["hessian"] = false);
  arma::vec result = out["par"];
  return(result);
}


arma::vec sv_sample_theta(arma::vec h, arma::vec theta){

  double mu, phi, phi_, sigma_eta, sigma_eta_;
  arma::vec x, grad;
  
  mu        = theta[0];
  phi       = theta[1];
  sigma_eta = theta[2];

  /*---Maximization---*/
  /* Reparameterize to remove the constraints */
  phi_      = log((1+phi)/(1-phi)); 
  sigma_eta_= log(sigma_eta);
  x         = {mu, phi_, sigma_eta_};

  arma::vec xhat, xn, xo;
  xo = x;
  x  = sv_Opt(x, h);

  xhat = x;
  /*---Calculation of Hessian Matrix---*/
  arma::mat hess = sv_deriv2(x, h);

  /*---Calculation of gradient vector---*/
  grad = sv_deriv1(x, h);
  
  /*---MH algorithm to sample vartheta---*/  
  arma::mat Q, Qinv, Qsqrt;
  Qinv  = -1.0*hess;
  Q     = arma::inv(Qinv);
  Qsqrt = arma::sqrtmat_sympd(Q);
  xn    = xhat + Q*grad + Qsqrt * Rcpp::as<arma::vec>(Rcpp::rnorm(3,0,1));
 
  double Lo, Ln, Lhat, qo, qn;
  Lo   = sv_theta_post_max(xo, h);
  Ln   = sv_theta_post_max(xn, h);
  Lhat = sv_theta_post_max(xhat, h);
  
  qo = arma::as_scalar( Lhat + grad.t()*(xo-xhat) - 0.5*(xo-xhat).t()*Qinv*(xo-xhat));
  qn = arma::as_scalar( Lhat + grad.t()*(xn-xhat) - 0.5*(xn-xhat).t()*Qinv*(xn-xhat));

  arma::vec vartheta;
  double frac = exp( Ln - Lo + qo - qn);
  double duni = Rcpp::as<double>(Rcpp::runif(1,0,1));
  if( duni < frac){ 
    vartheta = xn; cAccTheta = cAccTheta + 1; 
  }else{ 
    vartheta = xo; }

  mu         = vartheta[0]; 
  phi_       = vartheta[1];
  sigma_eta_ = vartheta[2];
  phi        = (exp(phi_)-1) / (exp(phi_)+1);
  sigma_eta  = exp(sigma_eta_);

  arma::vec result = {mu, phi, sigma_eta};
  return(result);
}
  
  arma::vec sv_sample_theta_pi1(arma::vec h, arma::vec theta, arma::vec theta_star){
    
    /********************************************************/
    /* We first sample theta as in asv_sample_theta below   */
    /********************************************************/
    double mu, phi, phi_, sigma_eta, sigma_eta_;
    arma::vec x, grad;
    
    mu        = theta[0];
    phi       = theta[1];
    sigma_eta = theta[2];

    
    /*---Maximization---*/
    phi_      = log((1+phi)/(1-phi)); // reparameterize to remove the constraint
    sigma_eta_= log(sigma_eta);
    x         = {mu, phi_, sigma_eta_};
    //
    arma::vec xhat, xn, xo;
    xo = x;
    x  = sv_Opt(x, h);
    //
    xhat = x;
    /*---Calculation of Hessian Matrix---*/
    arma::mat hess = sv_deriv2(x, h);
    
    /*---Calculation of gradient vector---*/
    grad = sv_deriv1(x, h);
    
    /*---MH algorithm to sample vartheta---*/  
    arma::mat Q, Qinv, Qsqrt;
    Qinv  = -1.0*hess;
    Q     = arma::inv(Qinv);
    Qsqrt = arma::sqrtmat_sympd(Q);
    xn    = xhat + Q*grad + Qsqrt * Rcpp::as<arma::vec>(Rcpp::rnorm(3,0,1));
    
    double Lo, Ln, Lhat, qo, qn;
    Lo   = sv_theta_post_max(xo, h);
    Ln   = sv_theta_post_max(xn, h);
    Lhat = sv_theta_post_max(xhat, h);
    
    qo = arma::as_scalar( Lhat + grad.t()*(xo-xhat) - 0.5*(xo-xhat).t()*Qinv*(xo-xhat));
    qn = arma::as_scalar( Lhat + grad.t()*(xn-xhat) - 0.5*(xn-xhat).t()*Qinv*(xn-xhat));
    
    arma::vec vartheta;
    double frac = exp( Ln - Lo + qo - qn);
    double duni = Rcpp::as<double>(Rcpp::runif(1,0,1));
    if( duni < frac){ 
      vartheta = xn; cAccTheta = cAccTheta + 1; 
    }else{ 
      vartheta = xo; }
    
    mu         = vartheta[0]; 
    phi_       = vartheta[1];
    sigma_eta_ = vartheta[2];
    phi        = (exp(phi_)-1) / (exp(phi_)+1);
    sigma_eta  = exp(sigma_eta_);

    /****************************************************/
    /* We finished sampling theta                       */
    /****************************************************/
    
    /****************************************************/
    /* Now we compute pi_1 below                        */
    /****************************************************/
    double mu_star, phi_star, phi_star_, sigma_eta_star, sigma_eta_star_;
    
    mu_star        = theta_star[0];
    phi_star       = theta_star[1];
    sigma_eta_star = theta_star[2];

    phi_star_       = log((1+phi_star)/(1-phi_star)); // reparameterize to remove the constraint
    sigma_eta_star_ = log(sigma_eta_star);
    arma::vec x_star   = {mu_star, phi_star_, sigma_eta_star_};
    
    /* Compute proposal density: q(theta, theta_star|y, h) = q(theta_star|y, h) 
     vartheta_star ~ N(xhat, Q) -> theta_star ~ N(xhat,Q)*jacobian  
     log proposal density : log N(xhat,Q) + log(jacobian)                     */
    
    double log_proposal;
    log_proposal  = -3.0*0.5*log(2.0*arma::datum::pi) -0.5*log(arma::det(Q));
    log_proposal += arma::as_scalar( -0.5*(x_star - xhat).t()*Qinv*(x_star - xhat) );
    log_proposal += - log(1-phi_star*phi_star) - 2.0*log(sigma_eta_star); 
    /* Compute acceptance probability: alpha(theta, theta_star|y, h) */
    Lo   = sv_theta_post_max(vartheta, h);
    Ln   = sv_theta_post_max(x_star, h);
    qo = arma::as_scalar( Lhat + grad.t()*(vartheta-xhat) - 0.5*(vartheta-xhat).t()*Qinv*(vartheta-xhat));
    qn = arma::as_scalar( Lhat + grad.t()*(x_star-xhat) - 0.5*(x_star-xhat).t()*Qinv*(x_star-xhat));
    
    double prob_acc = exp( Ln - Lo + qo - qn);
    if( prob_acc > 1.0 ){ prob_acc = 1.0; }
    double pi_1  = prob_acc * exp(log_proposal);
    
    arma::vec result = {mu, phi, sigma_eta, pi_1};
    
    return(result);
  }

double sv_sample_pi2(arma::vec h, arma::vec theta_star){
  
  /****************************************************/
  /* Replace theta by theta_star and                  */
  /*    generate theta as in asv_sample_theta below     */
  /****************************************************/
  double mu, phi, phi_, sigma_eta, sigma_eta_;
  arma::vec x, grad;
  
  /* replace theta by theta_star */
  mu        = theta_star[0];
  phi       = theta_star[1];
  sigma_eta = theta_star[2];
  /*                             */
  
  /*---Maximization---*/
  phi_      = log((1+phi)/(1-phi)); // reparameterize to remove the constraint
  sigma_eta_= log(sigma_eta);
  x         = {mu, phi_, sigma_eta_};
  //
  arma::vec xhat, xn, xo;
  xo = x;
  x  = sv_Opt(x, h);
  //
  xhat = x;
  /*---Calculation of Hessian Matrix---*/
  arma::mat hess = sv_deriv2(x, h);
  
  /*---Calculation of gradient vector---*/
  grad = sv_deriv1(x, h);
  
  /*---MH algorithm to sample vartheta---*/  
  arma::mat Q, Qinv, Qsqrt;
  Qinv  = -1.0*hess;
  Q     = arma::inv(Qinv);
  Qsqrt = arma::sqrtmat_sympd(Q);
  xn    = xhat + Q*grad + Qsqrt * Rcpp::as<arma::vec>(Rcpp::rnorm(3,0,1));
  
  double Lo, Ln, Lhat, qo, qn;
  Lo   = sv_theta_post_max(xo, h);
  Ln   = sv_theta_post_max(xn, h);
  Lhat = sv_theta_post_max(xhat, h);
  
  qo = arma::as_scalar( Lhat + grad.t()*(xo-xhat) - 0.5*(xo-xhat).t()*Qinv*(xo-xhat));
  qn = arma::as_scalar( Lhat + grad.t()*(xn-xhat) - 0.5*(xn-xhat).t()*Qinv*(xn-xhat));
  
  /****************************************************/
  /* Now compute the acceptance probability pi_2      */
  /****************************************************/
  double prob_acc = exp( Ln - Lo + qo - qn);
  if( prob_acc > 1.0 ){ prob_acc = 1.0; }
  
  double pi_2 = prob_acc;
  return(pi_2);
}


arma::vec sv_sample_h(arma::vec s, arma::vec h, arma::vec theta){

  arma::vec hn, ho;
  ho = h;
  hn = sv_sim_smoother(s,theta);
  
  /* MH algorithm */ 
  double mu, phi, sigma_eta;
  mu        = theta[0];
  phi       = theta[1];
  sigma_eta = theta[2];

  double dsum, hbarn, hbaro, lfn, lfo, gn, go;
  int i;
  
  dsum  = 0;
  
  int t;
  for(t = 0; t < T; t++){
    hbarn = mu*(1-phi) + phi*hn[t];
    hbaro = mu*(1-phi) + phi*ho[t];
    if( t < T-1 ){
      lfn = R::dnorm( Y[t], 0.0, exp(0.5*hn[t]), true)
          + R::dnorm( hn[t+1], hbarn, sigma_eta, true);
      lfo = R::dnorm( Y[t], 0.0, exp(0.5*ho[t]), true) 
          + R::dnorm( ho[t+1], hbaro, sigma_eta, true);
    }
    else{
      lfn = R::dnorm( Y[t], 0.0, exp(0.5*hn[t]), true);
      lfo = R::dnorm( Y[t], 0.0, exp(0.5*ho[t]), true); 
    }
    
    gn = 0; go = 0;
    if( t < T-1){
      for(i = 0; i < 10; i++){
        
        hbarn = mu*(1-phi) + phi*hn[t];
        hbaro = mu*(1-phi) + phi*ho[t];
        
        gn = gn + p[i]*R::dnorm( Y_star[t], hn[t]+m[i], v[i], false )
                      *R::dnorm( hn[t+1], hbarn, sigma_eta, false );            
        go = go + p[i]*R::dnorm( Y_star[t], ho[t]+m[i], v[i], false)
                      *R::dnorm( ho[t+1], hbaro, sigma_eta, false );            
      }
    }
    else{
      for(i = 0; i < 10; i++){
        gn = gn + p[i]*R::dnorm( Y_star[t], hn[t]+m[i], v[i], false );
        go = go + p[i]*R::dnorm( Y_star[t], ho[t]+m[i], v[i], false );
      }
    }    
    dsum = dsum + lfn - lfo + log(go) - log(gn);
  }
  
  double frac = exp(dsum);
  double uni  = Rcpp::as<double>(Rcpp::runif(1,0,1));
  arma::vec result;
  
  if( uni < frac){ 
    result = hn; cAccH = cAccH + 1;}
  else{ 
    result = ho; }
  
  return(result);  
}

double sv_pf(double mu, double phi, double sigma_eta, arma::vec Y, int I){
  double result = asv_pf(mu, phi, sigma_eta, 0, Y, I);
  return(result);
}

double sv_apf(double mu, double phi, double sigma_eta, arma::vec Y, int I){
  double result = asv_apf(mu, phi, sigma_eta, 0, Y, I);
  return(result);
}


// [[Rcpp::export]]
arma::vec sv_posterior(arma::vec H, arma::vec Theta, arma::vec Theta_star, arma::vec Y, 
                      Rcpp::Nullable<int> iM = R_NilValue,
                      Rcpp::Nullable<Rcpp::NumericVector> vHyper = R_NilValue){
  arma::vec h = H;
  arma::vec theta      = Theta;
  arma::vec theta_star = Theta_star;
  
  int L, M;
  /*---Setup---*/
  if(iM.isNotNull()){M = Rcpp::as<int>(iM);}   
  else{M = 5000;}
  
  L = M; 
  // L: # of MCMC iterations for pi_1
  // M: # of MCMC iterations for pi_2
  
  
  /*---Prior distributions for theta---*/
  // theta = {mu, phi, sigma_eta}
  //  mu_0    = -8.0; // mu~N(mu_0,sigma_0^2)
  if(vHyper.isNotNull())
  {
    Rcpp::NumericVector vhyper(vHyper);
    mu_0    = vhyper(0); // mu~N(mu_0,sigma_0^2)
    sigma_0 = vhyper(1);
    a_0     = vhyper(2);   // (phi+1)/2~Beta(a_0,b_0)
    b_0     = vhyper(3);
    n_0     = vhyper(4);    // sigma_eta^2~IG(n_0/2,S_0/2)
    S_0     = vhyper(5);
    
  }       
  else{
    mu_0    = 0.0; // mu~N(mu_0,sigma_0^2)
    sigma_0 = 1000;
    a_0     = 1.0;   // (phi+1)/2~Beta(a_0,b_0)
    b_0     = 1.0;
    n_0     = 0.01;    // sigma_eta^2~IG(n_0/2,S_0/2)
    S_0     = 0.01;
  }
  
  /*---Define variables---*/
  /*------Do not edit-----------------------------------*/ 
  // For the mixture approximation: 
  p       = {0.00609, 0.04775, 0.13057, 0.20674, 0.22715,
             0.18842, 0.12047, 0.05591, 0.01575, 0.00115};
  m       = {1.92677, 1.34744, 0.73504, 0.02266, -0.85173,
             -1.97278, -3.46788, -5.55246, -8.68384, -14.65};
  v2      = {0.11265, 0.17788, 0.26768, 0.40611, 0.62699,
             0.98583, 1.57469, 2.54498, 4.16591, 7.33342};
  v       = sqrt(v2); // above v is v^2
  a       = {1.01418, 1.02248, 1.03403, 1.05207, 1.08153,
             1.13114, 1.21754, 1.37454, 1.68327, 2.50097};
  b       = {0.5071, 0.51124, 0.51701, 0.52604, 0.54076,
             0.56557, 0.60877, 0.68728, 0.84163, 1.25049};
  dcst   = 1.0E-7;          
  /*----------------------------------------------------*/
  arma::vec s;
  Rcpp::NumericVector Y_, d_;
  
  Y_     = Rcpp::wrap(Y);   // # Declare Y as NumericVector of Rcpp
  d_     = Rcpp::ifelse(Y_ > 0, 1.0, -1.0); // T x 1 vector with t-th element =1 if y_t>0, -1 if y_t<0.
  d      = Rcpp::as<arma::vec>(d_); // Declare d_ as arma::vec
  Y_star = log(Y%Y + dcst); // log(y^2+c)
  
  
  arma::mat vpi_1  = arma::zeros(L);
  arma::vec vpi_2  = arma::zeros(M);
  
  arma::vec theta_pi1;
  
  /*---Start MCMC---*/
  Rprintf("Computing the log posterior density ...\n");
  Progress p2(L+M, true);
  int k;
  for(k = 0; k < L; k++){
    p2.increment();
    //    if(k % 500 == 0){Rprintf("Reduced MCMC iteration =%i\n", k);}
    /*---Sample theta---*/
    theta_pi1 = sv_sample_theta_pi1(h, theta, theta_star);
    theta     = theta_pi1(arma::span(0,2));
    vpi_1[k]  = theta_pi1[3];
    
    /*---Sample s ---*/
    s = sv_sample_s(h, theta);
    
    /*---Sample h_t ---*/
    h = sv_sample_h(s, h, theta);
  }
  
  /* Reduced run */
  //  Rprintf("Start Reduced MCMC run to compute the posterior\n");
  for(k = 0; k < M; k++){
    p2.increment();
    //  if(k % 500 == 0){Rprintf("Reduced MCMC iteration 2=%i\n", k);}
    /*---Sample theta---*/
    vpi_2[k] = sv_sample_pi2(h, theta_star);
    
    /*---Sample s with theta = theta_star---*/
    s = sv_sample_s(h, theta_star);
    
    /*---Sample h_t with theta = theta_star---*/
    h = sv_sample_h(s, h, theta_star);
  }
  
  /* Point estimate of the marginal likelihood */
  double pi_1, pi_2;
  pi_1 = arma::as_scalar( arma::mean(vpi_1) );
  pi_2 = arma::as_scalar( arma::mean(vpi_2) );
  double log_post   = log(pi_1) - log(pi_2);
  
  arma::vec b1, b2;
  arma::vec b_bar = arma::zeros(2);
  b_bar[0] = pi_1;
  b_bar[1] = pi_2;
  
  
  /*---Calculate standard err---*/
  arma::vec c_hat = {1/pi_1, -1/pi_2};
  int ss, ss_;
  int B_M = floor( 2*sqrt(M) + 1);
  arma::mat Omega, Omega_s; 
  Omega = arma::zeros(2,2);
  
  //  Rprintf("Compute the standard error of the log posterior\n");
  //  Progress p3(0.5*M*(M+1)-0.5*(M-B_M)*(M-B_M+1), true);
  for(ss = 0; ss < B_M; ss++){
    Omega_s = arma::zeros(2,2);
    b1      = arma::zeros(2);
    b2      = arma::zeros(2);
    for(ss_ = ss; ss_ < M; ss_++){
      //      p3.increment();
      b1[0] = vpi_1[ss_];
      b1[1] = vpi_2[ss_];
      b2[0] = vpi_1[ss_-ss];
      b2[1] = vpi_2[ss_-ss];
      Omega_s += ( b1 - b_bar) * ( b2 - b_bar).t() / M;
    }
    if(ss == 0){
      Omega += Omega_s / M;
    }else{
      Omega += 1 /M * (1 - ss / (B_M+1)) * (Omega_s + Omega_s.t());
    }
  }
  double std_err = sqrt(  arma::as_scalar( c_hat.t() * Omega * c_hat )  );
  //Rprintf("[Posterior density]standard err = %.4f\n", std_err);
  
  arma::vec result = {log_post, std_err};
  return(result);
}

// [[Rcpp::export]]
double sv_prior(arma::vec Theta_star, 
                 Rcpp::Nullable<Rcpp::NumericVector> vHyper = R_NilValue){
  
  arma::vec theta = Theta_star;
  
  double mu, phi, sigma_eta;
  mu        = theta[0];
  phi       = theta[1];
  sigma_eta = theta[2];

  
  /*---Prior distributions for theta---*/
  // theta = {mu, phi, sigma_eta, rho}
  if(vHyper.isNotNull())
  {
    Rcpp::NumericVector vhyper(vHyper);
    mu_0    = vhyper(0); // mu~N(mu_0,sigma_0^2)
    sigma_0 = vhyper(1);
    a_0     = vhyper(2);   // (phi+1)/2~Beta(a_0,b_0)
    b_0     = vhyper(3);
    n_0     = vhyper(4);    // sigma_eta^2~IG(n_0/2,S_0/2)
    S_0     = vhyper(5);
    
  }       
  else{
    mu_0    = 0.0; // mu~N(mu_0,sigma_0^2)
    sigma_0 = 1000;
    a_0     = 1.0;   // (phi+1)/2~Beta(a_0,b_0)
    b_0     = 1.0;
    n_0     = 0.01;    // sigma_eta^2~IG(n_0/2,S_0/2)
    S_0     = 0.01;
  }
  
  double log_prior_mu, log_prior_phi, log_prior_sigma_eta;
  log_prior_mu        = R::dnorm(mu, mu_0, sigma_0, true);
  log_prior_phi       = R::dbeta( (phi+1)/2, a_0, b_0, true ) - log(2);  
  log_prior_sigma_eta = R::dgamma(1/(sigma_eta*sigma_eta), n_0/2, 2/S_0, true)
    - 2 * log(sigma_eta*sigma_eta); 

  double log_prior = log_prior_mu + log_prior_phi + log_prior_sigma_eta;

  return(log_prior);
}


// [[Rcpp::export]]
Rcpp::List sv_logML(arma::vec H, arma::vec Theta, arma::vec Theta_star, arma::vec Y, 
                     Rcpp::Nullable<int> iI  = R_NilValue,
                     Rcpp::Nullable<int> iM  = R_NilValue,
                     Rcpp::Nullable<Rcpp::NumericVector> vHyper = R_NilValue){
  
  double mu, phi, sigma_eta;
  
  arma::vec theta_star = Theta_star;
  
  /* replace theta by theta_star */
  mu        = theta_star[0];
  phi       = theta_star[1];
  sigma_eta = theta_star[2];
  /*                             */
  
  /*---Setup---*/
  int I;
  if(iI.isNotNull()){I = Rcpp::as<int>(iI);}
  else{I = 5000;}
  
  /*---Compute Log Likelihood---*/
  Rprintf("Computing the log likelihood ...\n");
  int ir = 10;
  int k;
  arma::vec loglik = arma::zeros(ir);
  
  Progress p1(ir, true);
  for(k = 0; k < ir; k++){
    p1.increment();
    loglik[k] = sv_pf(mu, phi, sigma_eta, Y, I);
  }
  double loglik_m  = arma::mean(loglik);
  double loglik_se = sqrt(arma::var(loglik)/ir);
  Rprintf("Log likelihood ordinate (se) = %.4f (%.4f)\n", loglik_m, loglik_se);
  
  /*---Compute Log Prior---*/
  double logprior = sv_prior(Theta_star, vHyper);
  Rprintf("Log prior ordinate = %.4f\n", logprior);
  
  /*---Compute Log Posterior---*/
  arma::vec logpost = sv_posterior(H, Theta, Theta_star, Y, iM, vHyper);
  double logpost_m  = logpost[0];
  double logpost_se = logpost[1];
  
  Rprintf("Log posterior ordinate (se) = %.4f (%.4f)\n", logpost_m, logpost_se);
  
  double logML_m  = loglik_m + logprior - logpost_m;
  double logML_se = sqrt(loglik_se*loglik_se + logpost_se*logpost_se); 
  
  Rprintf("Log marginal likelihood (se) = %.4f (%.4f)\n", logML_m, logML_se);
  
  arma::vec vest = {logML_m,  loglik_m,  logprior, logpost_m };
  arma::vec vse  = {logML_se, loglik_se,        0, logpost_se};
  Rcpp::List result;
  result = Rcpp::List::create(vest, vse);  
  
  return(result);
}

//**********************************************//
// Stochastic volatility model with leverage    //
//**********************************************//
// [[Rcpp::export]]
Rcpp::List asv_mcmc(arma::vec return_vector, // Return vector
                  Rcpp::Nullable<double> nSim  = R_NilValue,
                  Rcpp::Nullable<double> nBurn = R_NilValue,
                  Rcpp::Nullable<Rcpp::NumericVector> vHyper = R_NilValue){
      
  /* Return vector */
  Y   = return_vector; 
  /* Sample size */
  T   = Y.n_elem;        // # of observations
  /*---Setup---*/
  double nsim, nburn;
  
  if(nSim.isNotNull()){nsim = Rcpp::as<double>(nSim);}       // # of MCMC iterations
  else{nsim = 5000;}
  
  if(nBurn.isNotNull()){nburn = Rcpp::as<double>(nBurn);}       
  else{nburn = 2.0 * sqrt(nsim) + 1 ;}// # of burn-in periods
  
  /*---Initial values---*/
  double phi, sigma_eta, mu, rho;
  // mu        = -8.00;
  mu        =  log( arma::sum(Y%Y)/T );
  phi       =  0.9;
  sigma_eta =  sqrt(0.3);
  rho       =  0.0;
  
  arma::vec h;
  
  h         = arma::zeros(T) + mu;
  int t;
  for(t = 1; t < T; t++){
    h[t] = mu + phi*(h[t-1] -mu) + R::rnorm(0, sigma_eta); 
  }
  
  /*---Prior distributions for theta---*/
  // theta = {mu, phi, sigma_eta, rho}
  //  mu_0    = -8.0; // mu~N(mu_0,sigma_0^2)
  if(vHyper.isNotNull())
  {
    Rcpp::NumericVector vhyper(vHyper);
    mu_0    = vhyper(0); // mu~N(mu_0,sigma_0^2)
    sigma_0 = vhyper(1);
    a_0     = vhyper(2);   // (phi+1)/2~Beta(a_0,b_0)
    b_0     = vhyper(3);
    a_1     = vhyper(4);   // (rho+1)/2~Beta(a_1,b_1)
    b_1     = vhyper(5);
    n_0     = vhyper(6);    // sigma_eta^2~IG(n_0/2,S_0/2)
    S_0     = vhyper(7);
    
  }       
  else{
    mu_0    = 0.0; // mu~N(mu_0,sigma_0^2)
    sigma_0 = 1000;
    a_0     = 1.0;   // (phi+1)/2~Beta(a_0,b_0)
    b_0     = 1.0;
    a_1     = 1.0;   // (rho+1)/2~Beta(a_1,b_1)
    b_1     = 1.0;
    n_0     = 0.01;    // sigma_eta^2~IG(n_0/2,S_0/2)
    S_0     = 0.01;
  }
  
  /*---Define variables---*/
  /*------Do not edit-----------------------------------*/ 
  // For the mixture approximation: 
  p       = {0.00609, 0.04775, 0.13057, 0.20674, 0.22715,
             0.18842, 0.12047, 0.05591, 0.01575, 0.00115};
  m       = {1.92677, 1.34744, 0.73504, 0.02266, -0.85173,
             -1.97278, -3.46788, -5.55246, -8.68384, -14.65};
  v2      = {0.11265, 0.17788, 0.26768, 0.40611, 0.62699,
             0.98583, 1.57469, 2.54498, 4.16591, 7.33342};
  v       = sqrt(v2); // above v is v^2
  a       = {1.01418, 1.02248, 1.03403, 1.05207, 1.08153,
             1.13114, 1.21754, 1.37454, 1.68327, 2.50097};
  b       = {0.5071, 0.51124, 0.51701, 0.52604, 0.54076,
             0.56557, 0.60877, 0.68728, 0.84163, 1.25049};
  dcst   = 1.0E-7;          
  /*----------------------------------------------------*/
  arma::vec s, theta;
  arma::vec mu_s, phi_s, sigma_eta_s, rho_s;
  Rcpp::NumericVector Y_, d_;
  Rcpp::List result;
  
  Y_     = Rcpp::wrap(Y);   // # Declare Y as NumericVector of Rcpp
  d_     = Rcpp::ifelse(Y_ > 0, 1.0, -1.0); // T x 1 vector with t-th element =1 if y_t>0, -1 if y_t<0.
  d      = Rcpp::as<arma::vec>(d_); // Declare d_ as arma::vec
  Y_star = log(Y%Y + dcst); // log(y^2+c)
  
  cAccH     = 0;cAccTheta = 0;
  
  theta = {mu, phi, sigma_eta, rho};
  
  mu_s        = arma::zeros(nsim);
  phi_s       = arma::zeros(nsim);
  sigma_eta_s = arma::zeros(nsim);
  rho_s       = arma::zeros(nsim);
  
  arma::mat h_s;
  h_s = arma::zeros(nsim, T);
  
  theta = asv_sample_theta(h, theta);
  s = asv_sample_s(h, theta);
  h = asv_sim_smoother(s,theta);
  
  /*---Start MCMC---*/
  Rprintf("Start MCMC estimation\n");
  Progress p1(nburn+nsim, true);
  int k;
  for(k = -nburn; k < nsim; k++){
    p1.increment();
//    if(k % 500 == 0){Rprintf("iteration=%i\n", k);}
    /* Reset counter */
    if(k == 0) { cAccH = 0; cAccTheta = 0; cRep = 0;}
    
    /*---Sample theta---*/
    theta = asv_sample_theta(h, theta);
    
    /*---Sample s ---*/
    s = asv_sample_s(h, theta);
    
    /*---Sample h_t ---*/
    h = asv_sample_h(s, h, theta);
    
    /*---Save MCMC samples---*/
    if(0 <= k){
      mu_s[k]        = theta[0];
      phi_s[k]       = theta[1];
      sigma_eta_s[k] = theta[2];
      rho_s[k]       = theta[3];
      h_s(k, arma::span(0,T-1)) = h.t();
    }
    cRep = cRep + 1;
  }
  Rprintf("Acceptance Rate for h     = %6.2f %%\n", cAccH/cRep*100);
  Rprintf("Acceptance Rate for (mu,phi,sigma_eta,rho) = %6.2f %%\n", cAccTheta/cRep*100);
  
  result = Rcpp::List::create(mu_s, phi_s, sigma_eta_s, rho_s, h_s);
  
  return(result);
}


arma::vec asv_sample_s(arma::vec h, arma::vec theta){
  
  double mu, phi, sigma_eta, rho;
  mu        = theta[0];
  phi       = theta[1];
  sigma_eta = theta[2];
  rho       = theta[3];
  
  int t;
  arma::vec eps_star, eta;
  arma::vec result = arma::zeros(T);
  for(t = 0; t < T; t++){
    
    /*---Calculate posterior density---*/
    eps_star = arma::pow((Y_star[t]-h[t]-m), 2) / (2*v%v); 
    // log likelihood 1 for (epsilon_star)
    if(t!=T-1){                                            
      // log likelihood 2 for (eta)
      eta = arma::pow((h[t+1]-mu)-phi*(h[t]-mu)-d[t]*rho*sigma_eta*arma::exp(m/2)%(a+b%(Y_star[t]-h[t]-m)), 2)
      /  (2*sigma_eta*sigma_eta*(1-rho*rho));
    }else{
      eta = arma::zeros(10);
    }
    
    arma::vec prob = p % (1/v) % (arma::exp(- (eps_star - arma::mean(eps_star))
                                              - (eta - arma::mean(eta))));
    prob           = prob / arma::sum(prob);
    Rcpp::NumericVector j = {0,1,2,3,4,5,6,7,8,9};
    double s_t   = Rcpp::sample(j, 1, true, Rcpp::wrap(prob))[0];
    result[t]    = s_t;
  }
  
  return(result);
}


Rcpp::List asv_kalman_filter(arma::vec s, arma::vec theta){
  
  double mu, phi, sigma_eta, rho;
  mu        = theta[0];
  phi       = theta[1];
  sigma_eta = theta[2];
  rho       = theta[3];
  
  //*---Kalman filter---*/
  int t;
  double h = mu;
  double P = sigma_eta*sigma_eta / (1-phi*phi);
  double h_star = 0;
  double A_star = -1;
  arma::vec e, D, K, L, f, F;
  arma::mat J;
  e = arma::zeros(T);
  D = arma::zeros(T);
  K = arma::zeros(T);
  L = arma::zeros(T);
  J = arma::zeros(T,2);
  f = arma::zeros(T);
  F = arma::zeros(T);
  
  for(t = 0; t < T; t++){
    arma::vec H = {d[t]*rho*sigma_eta*b[s[t]]*v[s[t]]*exp(m[s[t]]/2), sigma_eta*sqrt(1-rho*rho)};
    arma::vec G = {v[s[t]], 0};
    arma::vec W = {0, 1-phi, d[t]*rho*sigma_eta*a[s[t]]*exp(m[s[t]]/2)};
    arma::vec beta = {1, mu, 1};
    arma::vec B = {0, 1, 0};
    
    e[t] = Y_star[t] - m[s[t]] - h;
    D[t] = P + arma::dot(G,G);
    K[t] = (phi*P + arma::dot(H,G)) / D[t];
    L[t] = phi - K[t];
    J(t, arma::span(0,1)) = (H - K[t]*G).t();
    f[t] = Y_star[t] - m[s[t]] - h_star;
    F[t] = - A_star;
    
    h      = arma::dot(W,beta) + phi*h + K[t]*e[t];
    P      = phi*P*L[t] + arma::dot(H, J(t,arma::span(0,1)).t());
    h_star = W[2] + phi*h_star + K[t]*f[t];
    A_star = phi - 1 + phi*A_star + K[t]*F[t];
  }
  
  Rcpp::List result = Rcpp::List::create(e, D, J, L, f, F);
  
  return(result);
}


arma::vec asv_sim_smoother(arma::vec s, arma::vec theta){
  
  /*---(1) Kalman filter---*/
  int t;
  Rcpp::List kalman_list;
  
  double mu, phi, sigma_eta, rho;
  mu        = theta[0];
  phi       = theta[1];
  sigma_eta = theta[2];
  rho       = theta[3];
  
  kalman_list = asv_kalman_filter(s, theta);
  arma::vec e = kalman_list[0];
  arma::vec D = kalman_list[1];
  arma::mat J = kalman_list[2];
  arma::vec L = kalman_list[3];
  
  /*---(2) Simulation Smoother---*/
  double r, U, C, kappa, V;
  arma::vec eta, result;
  r = 0;
  U = 0;
  eta = arma::zeros(T);
  result = arma::ones(T) * mu;
  for(t = T-1; t > -1; t--){
    arma::mat H  = {d[t]*rho*sigma_eta*b[s[t]]*v[s[t]]*exp(m[s[t]]/2), sigma_eta*sqrt(1-rho*rho)};
    arma::mat G  = {v[s[t]], 0};
    arma::mat mJ = J(t,arma::span(0,1));
    arma::mat I  = arma::eye(2,2);
    
    arma::mat matC   = H * (I - G.t()*G/D[t] - mJ.t()*mJ*U) * H.t();
    C                = matC(0,0);
    kappa            = R::rnorm(0, sqrt(C));
    arma::mat matV   = H * (G.t()/D[t] + mJ.t()*U*L[t]);
    V                = matV(0,0);
    arma::mat mateta = H * (G.t()*e[t]/D[t] + mJ.t()*r) + kappa;
    eta[t]           = mateta(0,0);
    r                = e[t]/D[t] + L[t]*r - kappa*V/C;
    U                = 1/D[t] + L[t]*U*L[t] + V*V/C;
  }
  
  arma::mat H_0    = {0, sigma_eta/sqrt(1-phi*phi)};
  arma::mat mJ_0   = H_0;
  arma::mat I      = arma::eye(2,2);
  arma::mat matC_0 = H_0 * (I - mJ_0.t()*mJ_0*U) * H_0.t();
  double C_0       = arma::as_scalar(matC_0);
  double kappa_0   = R::rnorm(0, sqrt(C_0));
  arma::mat mateta = H_0 * mJ_0.t()*r + kappa_0;
  double eta_0     = arma::as_scalar(mateta);
  result[0]        = mu + eta_0;
  
  for(t = 1; t < T; t++){
    result[t] = mu*(1-phi)+d[t-1]*rho*sigma_eta*a[s[t-1]]*exp(m[s[t-1]]/2)
    +phi*result[t-1]+eta[t-1];
  }
  
  return(result);
}


double asv_loglikelihood(arma::vec s, arma::vec theta){
  
  /*---(1) Kalman filter---*/
  Rcpp::List kalman_list;
  kalman_list = asv_kalman_filter(s, theta);
  arma::vec D = kalman_list[1];
  arma::mat f = kalman_list[4];
  arma::vec F = kalman_list[5];
  
  /*---(2) Compute Log Likelihood---*/
  double C_1, mu_1;
  C_1  = 1 / ( 1/(sigma_0*sigma_0) + arma::sum(F%(1/D)%F) );
  mu_1 = C_1 * ( mu_0/(sigma_0*sigma_0) + arma::sum(F%(1/D)%f) );
  
  double result;
  // need to convert int T to double T?
  result = -T*log(2*arma::datum::pi)/2 - arma::sum(log(arma::abs(D)))/2 -
    log(fabs(sigma_0*sigma_0))/2 + log(fabs(C_1))/2 -
    ( arma::sum(f%(1/D)%f) + mu_0*(1/(sigma_0*sigma_0))*mu_0 - mu_1*(1/C_1)*mu_1 )/2;
  
  return(result);
}

double asv_loglikelihood_theta(arma::vec s, arma::vec theta){
  
  /*---(1) Kalman filter---*/
  Rcpp::List kalman_list;
  kalman_list = asv_kalman_filter(s, theta);
  arma::vec D = kalman_list[1];
  arma::mat f = kalman_list[4];
  arma::vec F = kalman_list[5];
  
  /*---(2) Compute Log Likelihood given theta =(mu, phi, sigma_eta, rho)---*/
  double result, mu;
  mu = theta[0];
  
  // need to convert int T to double T?
  result = -T*log(2*arma::datum::pi)/2 - arma::sum(log(arma::abs(D)))/2 
  - ( arma::sum(f%(1/D)%f) - 2 * mu* arma::sum(F%(1/D)%f) 
  + mu * mu *arma::sum(F%(1/D)%F)  )/2;
  
  return(result);
}


double asv_theta_post_max(arma::vec x, arma::vec h){
  // x={mu, phi_, sigma_eta_, rho_}
  
  double mu, phi_, sigma_eta_, rho_, phi, sigma_eta, rho;
  mu         = x[0];
  phi_       = x[1];
  phi        = (exp(phi_)-1) / (exp(phi_)+1);
  sigma_eta_ = x[2]; // = (1/2)*log(sigma_eta^2)
  sigma_eta  = exp(sigma_eta_);
  rho_       = x[3];
  rho        = (exp(rho_)-1) / (exp(rho_)+1);
  
  arma::vec theta = arma::zeros(4);
  theta[0] = mu;
  theta[1] = phi;
  theta[2] = sigma_eta;
  theta[3] = rho;
  
  //*---Compute Log Prior density---*/
  double log_prior_mu, log_prior_phi, log_prior_rho, log_prior_sigma_eta;
  log_prior_mu        = R::dnorm(mu, mu_0, sigma_0, true);
  log_prior_phi       = R::dbeta( (phi+1)/2, a_0, b_0, true ) - log(2);  // (phi+1)/2~Beta(a_0,b_0)
  log_prior_rho       = R::dbeta( (rho+1)/2, a_1, b_1, true ) - log(2);  // (rho+1)/2~Beta(a_1,b_1)  
  log_prior_sigma_eta = R::dgamma(1/(sigma_eta*sigma_eta), n_0/2, 2/S_0, true)
    - 2 * log(sigma_eta*sigma_eta); // sigma_eta^2~IG(n_0/2,S_0/2)
  
  /*---Compute Log Likelihood---*/
  double hbar, lf; // lf: log likelihood
  
  lf = R::dnorm( h[0], mu, sigma_eta/sqrt(1-phi*phi), true);
  int t;
  for(t = 0; t < T-1; t++){
    hbar = mu*(1-phi) + phi*h[t] + rho*sigma_eta*Y[t]*exp(-0.5*h[t]);
    if( t < T-1 ){
      lf = lf + R::dnorm( Y[t], 0.0, exp(0.5*h[t]), true )
      + R::dnorm( h[t+1], hbar, sigma_eta*sqrt(1-rho*rho), true);
    }
    else{
      lf = lf + R::dnorm( Y[t], 0.0, exp(0.5*h[t]), true );
    }
  }
  
  /*---Compute Jacobian---*/
  double jacobian = phi_ + log(2) + 2*sigma_eta_ + rho_ + 2*log(2)
    - 2*log(exp(phi_)+1) - 2*log(exp(rho_)+1);
  
  /*---Compute Log Posterior density---*/
  double result = lf + log_prior_mu + log_prior_phi + log_prior_rho  
                     + log_prior_sigma_eta + jacobian;
  return(result);
}


arma::vec asv_deriv1(arma::vec x, arma::vec h){
  // x={mu, phi_, sigma_eta_, rho_}
  int i;
  arma::vec e = arma::zeros(4);
  double epsilon = 0.001;
  arma::vec grad = arma::zeros(4);
  
  for(i = 0; i < 4; i++){
    e[i] = 1;
    grad[i] = ( 
      asv_theta_post_max(x+epsilon*e, h) - asv_theta_post_max(x-epsilon*e, h) 
    )/ (2*epsilon);
    e[i] = 0;
  }
  
  return(grad);
}


arma::mat asv_deriv2(arma::vec x, arma::vec h){
  // x={mu, phi, sigma_eta, rho}
  int i, j;
  arma::vec e_i, e_j;
  e_i = arma::zeros(4);
  e_j = arma::zeros(4);
  double epsilon = 0.000001;
  
  arma::mat hess = arma::zeros(4,4);
  
  for(i = 0; i < 4; i++){
    for(j = 0; j < 4; j++){
      e_i[i] = 1;
      e_j[j] = 1;
      hess(i,j) = (
        asv_theta_post_max(x+epsilon*e_i+epsilon*e_j, h)
        + asv_theta_post_max(x-epsilon*e_i-epsilon*e_j, h)
        - asv_theta_post_max(x-epsilon*e_i+epsilon*e_j, h)
        - asv_theta_post_max(x+epsilon*e_i-epsilon*e_j, h)
      ) / (4*epsilon*epsilon);
      e_i[i] = 0;
      e_j[j] = 0;
    }
  }
  
  return(hess);
}


arma::vec asv_Opt(arma::vec x, arma::vec h){
  // x={mu, phi_, sigma_eta_, rho}
  
  Rcpp::Environment stats("package:stats");
  Rcpp::Function optim = stats["optim"];
  
  Rcpp::List control = Rcpp::List::create(Rcpp::Named("fnscale") = -1.0);
  Rcpp::List out = optim(Rcpp::_["par"]    = x,
                         // Make sure this function is not exported!
                         Rcpp::_["fn"] = Rcpp::InternalFunction(&asv_theta_post_max),
                         Rcpp::_["h"] = h,
                         Rcpp::_["method"]  = "BFGS", 
                         Rcpp::_["control"] = control,
                         Rcpp::_["hessian"] = false);
  arma::vec result = out["par"];
  return(result);
}


arma::vec asv_sample_theta(arma::vec h, arma::vec theta){
  
  double mu, phi, phi_, sigma_eta, sigma_eta_, rho, rho_;
  arma::vec x, grad;
  
  mu        = theta[0];
  phi       = theta[1];
  sigma_eta = theta[2];
  rho       = theta[3];
  
  /*---Maximization---*/
  phi_      = log((1+phi)/(1-phi)); // reparameterize to remove the constraint
  sigma_eta_= log(sigma_eta);
  rho_      = log((1+rho)/(1-rho));
  x         = {mu, phi_, sigma_eta_, rho_};
  //
  arma::vec xhat, xn, xo;
  xo = x;
  x  = asv_Opt(x, h);
  //
  xhat = x;
  /*---Calculation of Hessian Matrix---*/
  arma::mat hess = asv_deriv2(x, h);
  
  /*---Calculation of gradient vector---*/
  grad = asv_deriv1(x, h);
  
  /*---MH algorithm to sample vartheta---*/  
  arma::mat Q, Qinv, Qsqrt;
  Qinv  = -1.0*hess;
  Q     = arma::inv(Qinv);
  Qsqrt = arma::sqrtmat_sympd(Q);
  xn    = xhat + Q*grad + Qsqrt * Rcpp::as<arma::vec>(Rcpp::rnorm(4,0,1));
  
  double Lo, Ln, Lhat, qo, qn;
  Lo   = asv_theta_post_max(xo, h);
  Ln   = asv_theta_post_max(xn, h);
  Lhat = asv_theta_post_max(xhat, h);
  
  qo = arma::as_scalar( Lhat + grad.t()*(xo-xhat) - 0.5*(xo-xhat).t()*Qinv*(xo-xhat));
  qn = arma::as_scalar( Lhat + grad.t()*(xn-xhat) - 0.5*(xn-xhat).t()*Qinv*(xn-xhat));
  
  arma::vec vartheta;
  double frac = exp( Ln - Lo + qo - qn);
  double duni = Rcpp::as<double>(Rcpp::runif(1,0,1));
  if( duni < frac){ 
    vartheta = xn; cAccTheta = cAccTheta + 1; 
  }else{ 
    vartheta = xo; }
  
  mu         = vartheta[0]; 
  phi_       = vartheta[1];
  sigma_eta_ = vartheta[2];
  rho_       = vartheta[3];
  phi        = (exp(phi_)-1) / (exp(phi_)+1);
  sigma_eta  = exp(sigma_eta_);
  rho        = (exp(rho_)-1) / (exp(rho_)+1);
  
  arma::vec result = {mu, phi, sigma_eta, rho};
  return(result);
}


arma::vec asv_sample_theta_pi1(arma::vec h, arma::vec theta, arma::vec theta_star){
  
  /********************************************************/
  /* We first sample theta as in asv_sample_theta below   */
  /********************************************************/
  double mu, phi, phi_, sigma_eta, sigma_eta_, rho, rho_;
  arma::vec x, grad;
  
  mu        = theta[0];
  phi       = theta[1];
  sigma_eta = theta[2];
  rho       = theta[3];
  

  /*---Maximization---*/
  phi_      = log((1+phi)/(1-phi)); // reparameterize to remove the constraint
  sigma_eta_= log(sigma_eta);
  rho_      = log((1+rho)/(1-rho));
  x         = {mu, phi_, sigma_eta_, rho_};
  //
  arma::vec xhat, xn, xo;
  xo = x;
  x  = asv_Opt(x, h);
  //
  xhat = x;
  /*---Calculation of Hessian Matrix---*/
  arma::mat hess = asv_deriv2(x, h);
  
  /*---Calculation of gradient vector---*/
  grad = asv_deriv1(x, h);
  
  /*---MH algorithm to sample vartheta---*/  
  arma::mat Q, Qinv, Qsqrt;
  Qinv  = -1.0*hess;
  Q     = arma::inv(Qinv);
  Qsqrt = arma::sqrtmat_sympd(Q);
  xn    = xhat + Q*grad + Qsqrt * Rcpp::as<arma::vec>(Rcpp::rnorm(4,0,1));
  
  double Lo, Ln, Lhat, qo, qn;
  Lo   = asv_theta_post_max(xo, h);
  Ln   = asv_theta_post_max(xn, h);
  Lhat = asv_theta_post_max(xhat, h);
  
  qo = arma::as_scalar( Lhat + grad.t()*(xo-xhat) - 0.5*(xo-xhat).t()*Qinv*(xo-xhat));
  qn = arma::as_scalar( Lhat + grad.t()*(xn-xhat) - 0.5*(xn-xhat).t()*Qinv*(xn-xhat));
  
  arma::vec vartheta;
  double frac = exp( Ln - Lo + qo - qn);
  double duni = Rcpp::as<double>(Rcpp::runif(1,0,1));
  if( duni < frac){ 
    vartheta = xn; cAccTheta = cAccTheta + 1; 
  }else{ 
    vartheta = xo; }
  
  mu         = vartheta[0]; 
  phi_       = vartheta[1];
  sigma_eta_ = vartheta[2];
  rho_       = vartheta[3];
  phi        = (exp(phi_)-1) / (exp(phi_)+1);
  sigma_eta  = exp(sigma_eta_);
  rho        = (exp(rho_)-1) / (exp(rho_)+1);
  
  /****************************************************/
  /* We finished sampling theta                       */
  /****************************************************/

  /****************************************************/
  /* Now we compute pi_1 below                        */
  /****************************************************/
  double mu_star, phi_star, phi_star_, sigma_eta_star, sigma_eta_star_, rho_star, rho_star_;

  mu_star        = theta_star[0];
  phi_star       = theta_star[1];
  sigma_eta_star = theta_star[2];
  rho_star       = theta_star[3];
  
  phi_star_       = log((1+phi_star)/(1-phi_star)); // reparameterize to remove the constraint
  sigma_eta_star_ = log(sigma_eta_star);
  rho_star_       = log((1+rho_star)/(1-rho_star));
  arma::vec x_star   = {mu_star, phi_star_, sigma_eta_star_, rho_star_};

  /* Compute proposal density: q(theta, theta_star|y, h) = q(theta_star|y, h) 
     vartheta_star ~ N(xhat, Q) -> theta_star ~ N(xhat,Q)*jacobian  
     log proposal density : log N(xhat,Q) + log(jacobian)                     */
  
  double log_proposal;
  log_proposal  = -4.0*0.5*log(2.0*arma::datum::pi) -0.5*log(arma::det(Q));
  log_proposal += arma::as_scalar( -0.5*(x_star - xhat).t()*Qinv*(x_star - xhat) );
  log_proposal += log(2.0) - log(1-phi_star*phi_star) - 2.0*log(sigma_eta_star) - log(1-rho_star*rho_star); 
  /* Compute acceptance probability: alpha(theta, theta_star|y, h) */
  Lo   = asv_theta_post_max(vartheta, h);
  Ln   = asv_theta_post_max(x_star, h);
  
  qo = arma::as_scalar( Lhat + grad.t()*(vartheta-xhat)   - 0.5*(vartheta-xhat).t()*Qinv*(vartheta-xhat));
  qn = arma::as_scalar( Lhat + grad.t()*(x_star-xhat) - 0.5*(x_star-xhat).t()*Qinv*(x_star-xhat));
  
  double prob_acc = exp( Ln - Lo + qo - qn);
  if( prob_acc > 1.0 ){ prob_acc = 1.0; }
  double pi_1  = prob_acc * exp(log_proposal);
  
  arma::vec result = {mu, phi, sigma_eta, rho, pi_1};
  
  return(result);
}

double asv_sample_pi2(arma::vec h, arma::vec theta_star){
  
  /****************************************************/
  /* Replace theta by theta_star and                  */
  /*    generate theta as in asv_sample_theta below     */
  /****************************************************/
  double mu, phi, phi_, sigma_eta, sigma_eta_, rho, rho_;
  arma::vec x, grad;
  
  /* replace theta by theta_star */
  mu        = theta_star[0];
  phi       = theta_star[1];
  sigma_eta = theta_star[2];
  rho       = theta_star[3];
  /*                             */

  /*---Maximization---*/
  phi_      = log((1+phi)/(1-phi)); // reparameterize to remove the constraint
  sigma_eta_= log(sigma_eta);
  rho_      = log((1+rho)/(1-rho));
  x         = {mu, phi_, sigma_eta_, rho_};
  //
  arma::vec xhat, xn, xo;
  xo = x;
  x  = asv_Opt(x, h);
  //
  xhat = x;
  /*---Calculation of Hessian Matrix---*/
  arma::mat hess = asv_deriv2(x, h);
  
  /*---Calculation of gradient vector---*/
  grad = asv_deriv1(x, h);
  
  /*---MH algorithm to sample vartheta---*/  
  arma::mat Q, Qinv, Qsqrt;
  Qinv  = -1.0*hess;
  Q     = arma::inv(Qinv);
  Qsqrt = arma::sqrtmat_sympd(Q);
  xn    = xhat + Q*grad + Qsqrt * Rcpp::as<arma::vec>(Rcpp::rnorm(4,0,1));
  
  double Lo, Ln, Lhat, qo, qn;
  Lo   = asv_theta_post_max(xo, h);
  Ln   = asv_theta_post_max(xn, h);
  Lhat = asv_theta_post_max(xhat, h);
  
  qo = arma::as_scalar( Lhat + grad.t()*(xo-xhat) - 0.5*(xo-xhat).t()*Qinv*(xo-xhat));
  qn = arma::as_scalar( Lhat + grad.t()*(xn-xhat) - 0.5*(xn-xhat).t()*Qinv*(xn-xhat));
  
  /****************************************************/
  /* Now compute the acceptance probability pi_2      */
  /****************************************************/
  double prob_acc = exp( Ln - Lo + qo - qn);
  if( prob_acc > 1.0 ){ prob_acc = 1.0; }

  double pi_2 = prob_acc;
  return(pi_2);
}


arma::vec asv_sample_h(arma::vec s, arma::vec h, arma::vec theta){
  
  arma::vec hn, ho;
  ho = h;
  hn = asv_sim_smoother(s,theta);
  
  /* MH algorithm */ 
  double mu, phi, sigma_eta, rho;
  mu        = theta[0];
  phi       = theta[1];
  sigma_eta = theta[2];
  rho       = theta[3];
  
  double dsum, hbarn, hbaro, lfn, lfo, gn, go;
  int i;
  
  dsum  = 0;
  
  int t;
  for(t = 0; t < T; t++){
    hbarn = mu*(1-phi) + phi*hn[t] + rho*sigma_eta*Y[t]*exp(-0.5*hn[t]);
    hbaro = mu*(1-phi) + phi*ho[t] + rho*sigma_eta*Y[t]*exp(-0.5*ho[t]);
    if( t < T-1 ){
      lfn = R::dnorm( Y[t], 0.0, exp(0.5*hn[t]), true )
      + R::dnorm( hn[t+1], hbarn, sigma_eta*sqrt(1-rho*rho), true);
      lfo = R::dnorm( Y[t], 0.0, exp(0.5*ho[t]), true ) 
        + R::dnorm( ho[t+1], hbaro, sigma_eta*sqrt(1-rho*rho), true);
    }
    else{
      lfn = R::dnorm( Y[t], 0.0, exp(0.5*hn[t]), true );
      lfo = R::dnorm( Y[t], 0.0, exp(0.5*ho[t]), true ); 
    }
    
    gn = 0; go = 0;
    if( t < T-1){
      for(i = 0; i < 10; i++){
        
        hbarn = mu*(1-phi) + phi*hn[t] + d[t]*rho*sigma_eta*exp(m[i]/2)
        * ( a[i] + b[i]*(Y_star[t]-hn[t]-m[i]) );
        hbaro = mu*(1-phi) + phi*ho[t] + d[t]*rho*sigma_eta*exp(m[i]/2)
          * ( a[i] + b[i]*(Y_star[t]-ho[t]-m[i]) );
        
        gn = gn + p[i]*R::dnorm( Y_star[t], hn[t]+m[i], v[i], false )
          *R::dnorm( hn[t+1], hbarn, sigma_eta*sqrt(1-rho*rho), false );            
        go = go + p[i]*R::dnorm( Y_star[t], ho[t]+m[i], v[i], false)
          *R::dnorm( ho[t+1], hbaro, sigma_eta*sqrt(1-rho*rho), false );            
      }
    }
    else{
      for(i = 0; i < 10; i++){
        gn = gn + p[i]*R::dnorm( Y_star[t], hn[t]+m[i], v[i], false );
        go = go + p[i]*R::dnorm( Y_star[t], ho[t]+m[i], v[i], false );
      }
    }    
    dsum = dsum + lfn - lfo + log(go) - log(gn);
  }
  
  double frac = exp(dsum);
  double uni  = Rcpp::as<double>(Rcpp::runif(1,0,1));
  arma::vec result;
  
  if( uni < frac){ 
    result = hn; cAccH = cAccH + 1;}
  else{ 
    result = ho; }
  
  return(result);  
}

arma::vec asv_mysample(arma::vec h, int I, arma::vec prob){
  
  Rcpp::Environment base("package:base");
  Rcpp::Function sample   = base["sample"];
  Rcpp::NumericVector out = sample(Rcpp::_["x"]    = h,
                                   // Make sure this function is not exported!
                                   Rcpp::_["size"] = I,
                                   Rcpp::_["replace"]  = true, 
                                   Rcpp::_["prob"] = prob);
  arma::vec result = Rcpp::as<arma::vec>(out);
  return(result);
}


double asv_pf(double mu, double phi, double sigma_eta, double rho, arma::vec Y, int I){
  
  int T   = Y.n_elem; // # of observations
  arma::vec  ws = arma::zeros(I);
  //  arma::vec  Ws = arma::zeros(I);
  arma::vec  w_means = arma::zeros(T);
  //  arma::vec  W_means = arma::zeros(T);

  /*---t=0---*/
  arma::vec h_0 = Rcpp::as<arma::vec>(Rcpp::rnorm(I, mu, sqrt(sigma_eta*sigma_eta/(1-phi*phi))));
  
  for(int i = 0; i < I; i++){
    ws[i] = R::dnorm(Y[0], 0, exp(h_0[i]/2), false);
//    Ws[i] = R::pnorm(Y[0], 0, exp(h_0[i]/2), true, false);
  }

  w_means[0] = arma::mean(ws);
//  W_means[0] = arma::mean(Ws);
  arma::vec f_hats = ws / arma::sum(ws);
  arma::vec h_ts = h_0;

  h_ts = reshape(Rcpp::RcppArmadillo::sample(h_ts, I, true, f_hats), I, 1);
  f_hats = arma::ones(I)/I;

  /*---1<=t---*/
  for(int t = 0; t < T-1; t++){
    arma::vec h_t_plus_1 = mu + phi * (h_ts - mu) + rho*sigma_eta*exp(-0.5*h_ts)*Y[t]
    + sqrt(1-rho*rho)*sigma_eta*(Rcpp::as<arma::vec>(Rcpp::rnorm(I, 0, 1)));
    for(int i = 0; i < I; i++){
      ws[i]  = R::dnorm(Y[t+1], 0, exp(h_t_plus_1[i]/2), false)*f_hats[i];
    //Ws[i]  = R::pnorm(Y[t+1], 0, exp(h_t_plus_1[i]/2), true, false)*f_hats[i];
    }
    
    w_means[t+1] = arma::sum(ws);
    //W_means[t+1] = arma::sum(Ws);
    f_hats = ws / arma::sum(ws);
    h_ts = h_t_plus_1;

    h_ts = reshape(Rcpp::RcppArmadillo::sample(h_t_plus_1, I, true, f_hats), I, 1);
    f_hats = arma::ones(I)/I;
  }
  double result = arma::sum(log(w_means));
  return(result);
}

double asv_apf(double mu, double phi, double sigma_eta, double rho, arma::vec Y, int I){
  int T   = Y.n_elem; // # of observations
  int i, k, t;
  arma::vec  ind     = arma::zeros(I);
  arma::vec  q       = arma::zeros(I);
  arma::vec  ws      = arma::zeros(I);
  //  arma::vec  Ws      = arma::zeros(I);
  arma::vec  w_means = arma::zeros(T);
  //  arma::vec  W_means = arma::zeros(T);
  
  /*---t=0---*/
  arma::vec h_0 = Rcpp::as<arma::vec>(Rcpp::rnorm(I, mu, sqrt(sigma_eta*sigma_eta/(1-phi*phi))));
  
  for(i = 0; i < I; i++){
    ws[i] = R::dnorm(Y[0], 0, exp(h_0[i]/2), false);
  //    Ws[i] = R::pnorm(Y[0], 0, exp(h_0[i]/2), true, false);
  }
  w_means[0] = arma::mean(ws);
  //  W_means[0] = arma::mean(Ws);
  arma::vec f_hats = ws / arma::sum(ws);

  arma::vec h_ts = h_0;
  //  h_ts = asv_mysample(h_ts, I, f_hats);
  h_ts   = reshape(Rcpp::RcppArmadillo::sample(h_0, I, true, f_hats), I, 1);
  f_hats = arma::ones(I)/I;
  
  /*---1<=t---*/
  for(t = 0; t < T-1; t++){
    arma::vec next_mus = mu + phi * (h_ts - mu) + rho*sigma_eta*exp(-h_ts/2)*Y[t];
    arma::vec g = 1 / sqrt(2*arma::datum::pi) * exp(-next_mus/2 - Y[t+1]*Y[t+1]*exp(-next_mus)/2) % f_hats
      / arma::sum(1 / sqrt(2*arma::datum::pi) * exp(-next_mus/2 - Y[t+1]*Y[t+1]*exp(-next_mus)/2) % f_hats);
    
    ind = arma::regspace(0, 1, I-1);
    // resample h_t ~ g using index
    ind = reshape(Rcpp::RcppArmadillo::sample(ind, I, true, g), I, 1);
    arma::vec h_t = h_ts; 
    arma::vec h_t_plus_1 = arma::zeros(I);// generate h_{t+1}|h_t,y_t
    for(i = 0; i < I; i++){
      k = ind[i]; 
      h_t[i] = h_ts[k];
      q[i]   = g[k];
      h_t_plus_1[i] = mu + phi * (h_t[i] - mu) + rho*sigma_eta*exp(-0.5*h_t[i])*Y[t]
      + R::rnorm(0, sqrt(1-rho*rho)*sigma_eta);
      
      ws[i]  = R::dnorm(Y[t+1], 0, exp(h_t_plus_1[i]/2), false) *f_hats[i]/ q[i];

    //  Ws[i]  = R::pnorm(Y[t+1], 0, exp(h_t_plus_1[i]/2), true, false)*f_hats[i] / q[i];
    }
    w_means[t+1] = arma::mean(ws);
    //  W_means[t+1] = arma::mean(Ws);
    f_hats       = ws / arma::sum(ws);

    h_ts = h_t_plus_1;
    //      h_ts = asv_mysample(h_t_plus_1, I, f_hats);
    h_ts = reshape(Rcpp::RcppArmadillo::sample(h_t_plus_1, I, true, f_hats), I, 1);
    f_hats = arma::ones(I)/I;
  }
  /*---Calculate estimated value of loglikelihood---*/
  double result = arma::sum(log(w_means));
  return(result);
}


// [[Rcpp::export]]
arma::vec asv_posterior(arma::vec H, arma::vec Theta, arma::vec Theta_star, arma::vec Y, 
                        Rcpp::Nullable<int> iM = R_NilValue,
                        Rcpp::Nullable<Rcpp::NumericVector> vHyper = R_NilValue){
  arma::vec h = H;
  arma::vec theta      = Theta;
  arma::vec theta_star = Theta_star;

  int L, M;
  /*---Setup---*/
  if(iM.isNotNull()){M = Rcpp::as<int>(iM);}   
  else{M = 5000;}
  
  L = M; 
  // L: # of MCMC iterations for pi_1
  // M: # of MCMC iterations for pi_2
  

  /*---Prior distributions for theta---*/
  // theta = {mu, phi, sigma_eta, rho}
  //  mu_0    = -8.0; // mu~N(mu_0,sigma_0^2)
  if(vHyper.isNotNull())
  {
    Rcpp::NumericVector vhyper(vHyper);
    mu_0    = vhyper(0); // mu~N(mu_0,sigma_0^2)
    sigma_0 = vhyper(1);
    a_0     = vhyper(2);   // (phi+1)/2~Beta(a_0,b_0)
    b_0     = vhyper(3);
    a_1     = vhyper(4);   // (rho+1)/2~Beta(a_1,b_1)
    b_1     = vhyper(5);
    n_0     = vhyper(6);    // sigma_eta^2~IG(n_0/2,S_0/2)
    S_0     = vhyper(7);
    
  }       
  else{
    mu_0    = 0.0; // mu~N(mu_0,sigma_0^2)
    sigma_0 = 1000;
    a_0     = 1.0;   // (phi+1)/2~Beta(a_0,b_0)
    b_0     = 1.0;
    a_1     = 1.0;   // (rho+1)/2~Beta(a_1,b_1)
    b_1     = 1.0;
    n_0     = 0.01;    // sigma_eta^2~IG(n_0/2,S_0/2)
    S_0     = 0.01;
  }
  
  /*---Define variables---*/
  /*------Do not edit-----------------------------------*/ 
  // For the mixture approximation: 
  p       = {0.00609, 0.04775, 0.13057, 0.20674, 0.22715,
             0.18842, 0.12047, 0.05591, 0.01575, 0.00115};
  m       = {1.92677, 1.34744, 0.73504, 0.02266, -0.85173,
             -1.97278, -3.46788, -5.55246, -8.68384, -14.65};
  v2      = {0.11265, 0.17788, 0.26768, 0.40611, 0.62699,
             0.98583, 1.57469, 2.54498, 4.16591, 7.33342};
  v       = sqrt(v2); // above v is v^2
  a       = {1.01418, 1.02248, 1.03403, 1.05207, 1.08153,
             1.13114, 1.21754, 1.37454, 1.68327, 2.50097};
  b       = {0.5071, 0.51124, 0.51701, 0.52604, 0.54076,
             0.56557, 0.60877, 0.68728, 0.84163, 1.25049};
  dcst   = 1.0E-7;          
  /*----------------------------------------------------*/
  arma::vec s;
  Rcpp::NumericVector Y_, d_;

  Y_     = Rcpp::wrap(Y);   // # Declare Y as NumericVector of Rcpp
  d_     = Rcpp::ifelse(Y_ > 0, 1.0, -1.0); // T x 1 vector with t-th element =1 if y_t>0, -1 if y_t<0.
  d      = Rcpp::as<arma::vec>(d_); // Declare d_ as arma::vec
  Y_star = log(Y%Y + dcst); // log(y^2+c)
  
  
  arma::mat vpi_1  = arma::zeros(L);
  arma::vec vpi_2  = arma::zeros(M);

  arma::vec theta_pi1;
  
  /*---Start MCMC---*/
  Rprintf("Computing the log posterior density ...\n");
  Progress p2(L+M, true);
  int k;
  for(k = 0; k < L; k++){
    p2.increment();
//    if(k % 500 == 0){Rprintf("Reduced MCMC iteration =%i\n", k);}
    /*---Sample theta---*/
    theta_pi1 = asv_sample_theta_pi1(h, theta, theta_star);
    theta     = theta_pi1(arma::span(0,3));
    vpi_1[k]  = theta_pi1[4];
    
    /*---Sample s ---*/
    s = asv_sample_s(h, theta);
    
    /*---Sample h_t ---*/
    h = asv_sample_h(s, h, theta);
  }

  /* Reduced run */
//  Rprintf("Start Reduced MCMC run to compute the posterior\n");
  for(k = 0; k < M; k++){
    p2.increment();
    //  if(k % 500 == 0){Rprintf("Reduced MCMC iteration 2=%i\n", k);}
    /*---Sample theta---*/
    vpi_2[k] = asv_sample_pi2(h, theta_star);
    
    /*---Sample s with theta = theta_star---*/
    s = asv_sample_s(h, theta_star);
    
    /*---Sample h_t with theta = theta_star---*/
    h = asv_sample_h(s, h, theta_star);
  }

  /* Point estimate of the marginal likelihood */
  double pi_1, pi_2;
  pi_1 = arma::as_scalar( arma::mean(vpi_1) );
  pi_2 = arma::as_scalar( arma::mean(vpi_2) );
  double log_post   = log(pi_1) - log(pi_2);
  
  arma::vec b1, b2;
  arma::vec b_bar = arma::zeros(2);
  b_bar[0] = pi_1;
  b_bar[1] = pi_2;
  
  
  /*---Calculate standard err---*/
  arma::vec c_hat = {1/pi_1, -1/pi_2};
  int ss, ss_;
  int B_M = floor( 2*sqrt(M) + 1);
  arma::mat Omega, Omega_s; 
  Omega = arma::zeros(2,2);
  
//  Rprintf("Compute the standard error of the log posterior\n");
//  Progress p3(0.5*M*(M+1)-0.5*(M-B_M)*(M-B_M+1), true);
  for(ss = 0; ss < B_M; ss++){
    Omega_s = arma::zeros(2,2);
    b1      = arma::zeros(2);
    b2      = arma::zeros(2);
    for(ss_ = ss; ss_ < M; ss_++){
//      p3.increment();
      b1[0] = vpi_1[ss_];
	    b1[1] = vpi_2[ss_];
	    b2[0] = vpi_1[ss_-ss];
	    b2[1] = vpi_2[ss_-ss];
      Omega_s += ( b1 - b_bar) * ( b2 - b_bar).t() / M;
    }
    if(ss == 0){
      Omega += Omega_s / M;
    }else{
      Omega += 1 /M * (1 - ss / (B_M+1)) * (Omega_s + Omega_s.t());
    }
  }
  double std_err = sqrt(  arma::as_scalar( c_hat.t() * Omega * c_hat )  );
  //Rprintf("[Posterior density]standard err = %.4f\n", std_err);
  
  arma::vec result = {log_post, std_err};
  return(result);
}

// [[Rcpp::export]]
double asv_prior(arma::vec Theta_star, 
                 Rcpp::Nullable<Rcpp::NumericVector> vHyper = R_NilValue){

  arma::vec theta = Theta_star;
  
  double mu, phi, sigma_eta, rho;
  mu        = theta[0];
  phi       = theta[1];
  sigma_eta = theta[2];
  rho       = theta[3];
  
  
  /*---Prior distributions for theta---*/
  // theta = {mu, phi, sigma_eta, rho}
  if(vHyper.isNotNull())
  {
    Rcpp::NumericVector vhyper(vHyper);
    mu_0    = vhyper(0); // mu~N(mu_0,sigma_0^2)
    sigma_0 = vhyper(1);
    a_0     = vhyper(2);   // (phi+1)/2~Beta(a_0,b_0)
    b_0     = vhyper(3);
    a_1     = vhyper(4);   // (rho+1)/2~Beta(a_1,b_1)
    b_1     = vhyper(5);
    n_0     = vhyper(6);    // sigma_eta^2~IG(n_0/2,S_0/2)
    S_0     = vhyper(7);
    
  }       
  else{
    mu_0    = 0.0; // mu~N(mu_0,sigma_0^2)
    sigma_0 = 1000;
    a_0     = 1.0;   // (phi+1)/2~Beta(a_0,b_0)
    b_0     = 1.0;
    a_1     = 1.0;   // (rho+1)/2~Beta(a_1,b_1)
    b_1     = 1.0;
    n_0     = 0.01;    // sigma_eta^2~IG(n_0/2,S_0/2)
    S_0     = 0.01;
  }
  
  double log_prior_mu, log_prior_phi, log_prior_sigma_eta, log_prior_rho;
  log_prior_mu        = R::dnorm(mu, mu_0, sigma_0, true);
  log_prior_phi       = R::dbeta( (phi+1)/2, a_0, b_0, true ) - log(2);  
  log_prior_sigma_eta = R::dgamma(1/(sigma_eta*sigma_eta), n_0/2, 2/S_0, true)
                        - 2 * log(sigma_eta*sigma_eta); 
  log_prior_rho       = R::dbeta( (rho+1)/2, a_1, b_1, true ) - log(2);  

  double log_prior = log_prior_mu + log_prior_phi + log_prior_sigma_eta
                   + log_prior_rho;
  
  return(log_prior);
}


// [[Rcpp::export]]
Rcpp::List asv_logML(arma::vec H, arma::vec Theta, arma::vec Theta_star, arma::vec Y, 
                     Rcpp::Nullable<int> iI  = R_NilValue,
                     Rcpp::Nullable<int> iM  = R_NilValue,
                     Rcpp::Nullable<Rcpp::NumericVector> vHyper = R_NilValue){
  
  double mu, phi, sigma_eta, rho;
  
  arma::vec theta_star = Theta_star;
  
  /* replace theta by theta_star */
  mu        = theta_star[0];
  phi       = theta_star[1];
  sigma_eta = theta_star[2];
  rho       = theta_star[3];
  /*                             */
  
  /*---Setup---*/
  int I;
  if(iI.isNotNull()){I = Rcpp::as<int>(iI);}
  else{I = 5000;}

  /*---Compute Log Likelihood---*/
  Rprintf("Computing the log likelihood ...\n");
  int ir = 10;
  int k;
  arma::vec loglik = arma::zeros(ir);
  
  Progress p1(ir, true);
  for(k = 0; k < ir; k++){
    p1.increment();
    loglik[k] = asv_pf(mu, phi, sigma_eta, rho, Y, I);
  }
  double loglik_m  = arma::mean(loglik);
  double loglik_se = sqrt(arma::var(loglik)/ir);
  Rprintf("Log likelihood ordinate (se) = %.4f (%.4f)\n", loglik_m, loglik_se);

  /*---Compute Log Prior---*/
  double logprior = asv_prior(Theta_star, vHyper);
  Rprintf("Log prior ordinate = %.4f\n", logprior);
  
  /*---Compute Log Posterior---*/
  arma::vec logpost = asv_posterior(H, Theta, Theta_star, Y, iM, vHyper);
  double logpost_m  = logpost[0];
  double logpost_se = logpost[1];
  
  Rprintf("Log posterior ordinate (se) = %.4f (%.4f)\n", logpost_m, logpost_se);

  double logML_m  = loglik_m + logprior - logpost_m;
  double logML_se = sqrt(loglik_se*loglik_se + logpost_se*logpost_se); 
    
  Rprintf("Log marginal likelihood (se) = %.4f (%.4f)\n", logML_m, logML_se);
    
  arma::vec vest = {logML_m,  loglik_m,  logprior, logpost_m };
  arma::vec vse  = {logML_se, loglik_se,        0, logpost_se};
  Rcpp::List result;
  result = Rcpp::List::create(vest, vse);  

  return(result);
}


