/*!
 * \file
 * \brief   Lab 1 - Threading.
 * \author  Johann A. Briffa
 *
 * Template for the solution to Lab 1 practical exercise on Monte Carlo
 * integration.
 */

#include "../shared/jbutil.h"
#include <thread>
#include <vector>

// Define the integration limits
#define a -2
#define b 2

// Define the function limits 
#define A 0
#define B 0.4

// Define mean and variance 
#define mean 0
#define variance 1

// Define number of threads
#define num_threads 4

//Defining a struct for the results 
struct Results
{
   int count;
   double integral_estimate;
   double error;
};

// Monte Carlo integration function
Results MonteCarlo(const int N)
   {
   std::cerr << "\nImplementation (" << N << " samples)" << std::endl;
   // start timer
   double t = jbutil::gettime();

   // define function to integrate
   // f(x) = (1/sqrt(2.pi.variance^2)) * exp(-(x-mean)^2/2.variance^2)
   auto f = [](double x) -> double
   {
      return (1.0 / sqrt(2.0 * pi * variance * variance)) * exp(-pow(x - mean, 2) / (2.0 * variance * variance));
   };
   
   // define random number generator
   jbutil::randgen rng;

   // initialise count for samples below the function
   int count = 0;

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
   }

   // Integral estimate
   double integral_estimate = ( (count/(double)N) * (b - a) * (B - A) ) + ( A* (b-a) );

   // error function for integral estimate
   double error = erf(sqrt(2));

   // stop timer
   t = jbutil::gettime() - t;
   std::cerr << "Time taken: " << t << "s" << std::endl;

   return {count, integral_estimate, error};
   }

// Threaded Monte Carlo integration function
void ThreadedMonteCarlo(const int N)
   {
      // define number of samples per thread
      const int samples_per_thread = N / num_threads;

      // declare vector of threads
      std::vector<std::thread> threads;

      // iterate over number of threads
      for (int i = 0; i < num_threads; ++i)
      {
         // create thread
         threads.push_back(std::thread([samples_per_thread](){
            // call Monte Carlo function
            MonteCarlo(samples_per_thread);
         }));
      }

      // iterate over threads to join them together
      for (auto& thread : threads)
      {
         // wait for thread to finish
         thread.join();
      }
      
   }

// Main program entry point
int main()
   {
   std::cerr << "Lab 1: Monte Carlo integration" << std::endl;
   const int N = int(1E8);
   // save results of Monte Carlo function
   Results results = MonteCarlo(N);

   // print results
   std::cerr << "Samples: " << N << std::endl;
   std::cerr << "Samples below the function: " << results.count << std::endl;
   std::cerr << "--------------------------------" << std::endl;
   std::cerr << "Integral Estimate: " << results.integral_estimate << std::endl;
   std::cerr << "Error Estimate: " << results.error << std::endl;
   }
