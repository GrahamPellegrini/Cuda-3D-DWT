/*!
 * \file
 * \brief   Lab 1 - Threading.
 * \author  Johann A. Briffa
 *
 * Template for the solution to Lab 1 practical exercise on Monte Carlo
 * integration.
 */

#include "../shared/jbutil.h"

// Monte Carlo integration function

void MonteCarlo(const int N)
   {
   std::cerr << "\nImplementation (" << N << " samples)" << std::endl;
   // start timer
   double t = jbutil::gettime();

   // define integration limits
   const double a = -2;
   const double b = 2;

   // define y_i limits for function
   const double A = 0;
   const double B = 0.4;

   // define function to integrate
   // f(x) = (1/sqrt(2.pi.variance^2)) * exp(-(x-mean)^2/2.variance^2)
   auto f = [](double x) -> double
   {
      const double mean = 0;
      const double variance = 1;
      return (1.0 / sqrt(2.0 * pi * variance * variance)) * exp(-pow(x - mean, 2) / (2.0 * variance * variance));
   };
   
   // define random number generator
   jbutil::randgen rng;

   // initialise count for samples below the function
   int count = 0;

   // initialise sum for integral estimate
   double sum = 0;

   // iterate over samples
   for(int i = 0; i < N; ++i)
   {
      // generate random number for x_i between a and b
      double x = rng.fval(a,b);

      // generate random number for y_i between 0 and 1
      double y = rng.fval(A,B);

      // evaluate function at x_i
      double f_x = f(x);

      // check if y_i is below the function
      if(y < f_x)
      {
         // if so, increment the count
         count++;
      }

      // calculate the integral
      sum += f_x;
   }

   // Integral estimate
   double integral_estimate = ( (count/(double)N) * (b - a) * (B - A) ) + ( A* (b-a) );

   // error function for integral estimate
   double error = erf(sqrt(2));

   // print results
   std::cerr << "Samples: " << N << std::endl;
   std::cerr << "Samples below the function: " << count << std::endl;
   std::cerr << "--------------------------------" << std::endl;
   std::cerr << "Monte Carlo Fucntion Integration" << sum << std::endl;
   std::cerr << "Integral Estimate: " << integral_estimate << std::endl;
   std::cerr << "Error: " << error << std::endl;

   // stop timer
   t = jbutil::gettime() - t;
   std::cerr << "Time taken: " << t << "s" << std::endl;
   }

// Main program entry point

int main()
   {
   std::cerr << "Lab 1: Monte Carlo integration" << std::endl;
   const int N = int(1E8);
   MonteCarlo(N);
   }
