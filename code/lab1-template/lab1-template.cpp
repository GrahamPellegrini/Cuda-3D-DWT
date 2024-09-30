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

   // TODO: write your implementation here

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
