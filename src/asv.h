// asv.h
// written by 
// Ryuji Hashimoto and Yasuhiro Omori (University of Tokyo)
//

//******* Global variables: fixed constants ******//
/* For mixture approximation */
arma::vec p, m, v, v2, a, b;

/* offset value (very small value c for y*_t=log(y_t^2+c))*/
double   dcst;

/* For hyper-parameters for prior distributions */
double a_0, b_0, a_1, b_1, n_0, S_0, mu_0, sigma_0;

/* For dependent variables */
//   y_t* = log(y_t^2 + small const)
//   d_t = 1 if y_t* > 0, d_t = -1 o.w.
arma::vec Y, Y_star, d; 

/* Sample size */
int T;
//******* Global variables***********************//
 /* Compute the acceptance rate in MH algorithm */
double cAccH, cAccTheta, cRep;
//*********************************************//

//**********************************************//
// Stochastic volatility model without leverage //
//**********************************************//

arma::vec sv_sample_s(arma::vec h, arma::vec theta);

Rcpp::List sv_kalman_filter(arma::vec s, arma::vec theta);

arma::vec sv_sim_smoother(arma::vec s, arma::vec theta);

double sv_loglikelihood(arma::vec s, arma::vec theta);

double sv_loglikelihood_theta(arma::vec s, arma::vec theta);

double sv_theta_post_max(arma::vec x, arma::vec h);

arma::vec sv_deriv1(arma::vec x, arma::vec h);

arma::mat sv_deriv2(arma::vec x, arma::vec h);

arma::vec sv_Opt(arma::vec x, arma::vec h);

arma::vec sv_sample_theta(arma::vec h, arma::vec theta);

arma::vec sv_sample_h(arma::vec s, arma::vec h, arma::vec theta);

// [[Rcpp::export]]
double sv_pf(double mu, double phi, double sigma_eta, arma::vec Y,  int I);

// [[Rcpp::export]]
double sv_apf(double mu, double phi, double sigma_eta, arma::vec Y, int I);

//**********************************************//
// Stochastic volatility model with leverage    //
//**********************************************//

arma::vec asv_sample_s(arma::vec h, arma::vec theta);

Rcpp::List asv_kalman_filter(arma::vec s, arma::vec theta);

arma::vec asv_sim_smoother(arma::vec s, arma::vec theta);

double asv_loglikelihood(arma::vec s, arma::vec theta);

double asv_loglikelihood_theta(arma::vec s, arma::vec theta);

double asv_theta_post_max(arma::vec x, arma::vec h);

arma::vec asv_deriv1(arma::vec x, arma::vec h);

arma::mat asv_deriv2(arma::vec x, arma::vec h);

arma::vec asv_Opt(arma::vec x, arma::vec h);

arma::vec asv_sample_theta(arma::vec h, arma::vec theta);

arma::vec asv_sample_h(arma::vec s, arma::vec h, arma::vec theta);

arma::vec asv_mysample(arma::vec h, int I, arma::vec prob);

// [[Rcpp::export]]
double asv_pf(double mu, double phi, double sigma_eta, double rho, arma::vec Y,  int I);

// [[Rcpp::export]]
double asv_apf(double mu, double phi, double sigma_eta, double rho, arma::vec Y,  int I);


  
