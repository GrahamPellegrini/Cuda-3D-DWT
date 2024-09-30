/*!
 * \file
 * \brief   Lab 2 - SIMD Programming.
 * \author  Johann A. Briffa
 *
 * Template for the solution to Lab 2 practical exercise on image resampling
 * using the Lanczos filter.
 */

#include "../shared/jbutil.h"
#include <cmath>

// Define the sinc function
double sinc(double x)
{
   if (x == 0.0)
      return 1.0;
   return sin(pi * x) / (pi * x);
}

// Define the Lanczos filter
double lanczos(double x, int a)
{
   // If x is zero, return 1
   if (x == 0.0)
      return 1.0;
   // If x absolute value is greater than or equal to a, return 0
   if (fabs(x) >= a)
      return 0.0;
   // Otherwise, return the Lanczos filter function
   return sinc(x) * sinc(x / a);
}

// Define the Lanczos 2D filter
double lanczos_2d(double x, double y, int a)
{
   return lanczos(x, a) * lanczos(y, a);
}

// Resample the image using Lanczos filter
template <class real>
void process(const std::string infile, const std::string outfile,
      const real R, const int a)
{
   // load image
   jbutil::image<int> image_in;
   std::ifstream file_in(infile.c_str());
   image_in.load(file_in);

   // Debugging information to make sure the image was loaded correctly
   std::cerr << "Image loaded: " << image_in.get_rows() << " rows, " 
          << image_in.get_cols() << " columns, " 
          << image_in.channels() << " channels." << std::endl;


   // get input image dimensions (cols, rows)
   int rows_in = image_in.get_rows();
   int cols_in = image_in.get_cols();

   // start timer
   double t = jbutil::gettime();  

   // calculate output image dimensions depending on the scale factor
   int rows_out = static_cast<int>(rows_in * R);
   int cols_out = static_cast<int>(cols_in * R);
   // Note the int cast due to R factor being a float and the output dimensions being integers

   // create output image
   jbutil::image<int> image_out(rows_in, cols_out);
   
   // Print line 
   std::cerr << "Resampling image to " << rows_out << " rows, " << cols_out << " columns." << std::endl;

   // Lanczos resampling
   for (int m = 0; m < rows_out; ++m)
   {
      for (int n = 0; n < cols_out; ++n)
      {
         
         // map output pixel (m, n) to input pixel space
         double mR = m / R;
         double nR = n / R;


      }
   } 
   // stop timer
   t = jbutil::gettime() - t;

   // save output image
   std::ofstream file_out(outfile.c_str());
   image_out.save(file_out);

   // show time taken
   std::cerr << "Time taken: " << t << "s" << std::endl;
}

// Main program entry point
int main(int argc, char *argv[])
{
   std::cerr << "Lab 2: Image resampling with Lanczos filter" << std::endl;
   if (argc != 5)
   {
      std::cerr << "Usage: " << argv[0]
            << " <infile> <outfile> <scale-factor> <limit>" << std::endl;
      exit(1);
   }
   process<float> (argv[1], argv[2], atof(argv[3]), atoi(argv[4]));
}
