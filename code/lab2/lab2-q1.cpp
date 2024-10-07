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


// Clamp function to keep pixel values in the valid range
template <class T>
T clamp(T value, T min_val, T max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

// Resample the image using Lanczos filter
template <class real>
void process(const std::string infile, const std::string outfile,
             const real R, const int a)
{
    // Load image
    jbutil::image<int> image_in;
    std::ifstream file_in(infile.c_str());
    image_in.load(file_in);

    std::cerr << "Image loaded: " << image_in.get_rows() << " rows, "
              << image_in.get_cols() << " columns, "
              << image_in.channels() << " channels." << std::endl;

    // Get input image dimensions
    int rows_in = image_in.get_rows();
    int cols_in = image_in.get_cols();

    // Start timer
    double t = jbutil::gettime();

    // Calculate output image dimensions based on scale factor
    int rows_out = static_cast<int>(rows_in * R);
    int cols_out = static_cast<int>(cols_in * R);

    std::cerr << "Resampling image to " << rows_out << " rows, " << cols_out << " columns." << std::endl;

    // Create output image
    jbutil::image<int> image_out(rows_out, cols_out, image_in.channels());

    // Lanczos resampling
    for (int m = 0; m < rows_out; ++m)
    {
        for (int n = 0; n < cols_out; ++n)
        {
            // Map output pixel (m, n) to input pixel space
            double mR = m / R;
            double nR = n / R;

            // Initialize output pixel value
            double sum = 0.0;

            // Sum the Lanczos contributions over a window of 'a' around the mapped pixel
            for (int i = -a + 1; i <= a; ++i)
            {
                for (int j = -a + 1; j <= a; ++j)
                {
                    // Coordinates of neighboring pixel in the input image
                    int m_in = static_cast<int>(mR + i);
                    int n_in = static_cast<int>(nR + j);

                    // Make sure we're inside the image bounds
                    if (m_in >= 0 && m_in < rows_in && n_in >= 0 && n_in < cols_in)
                    {
                        // Apply the Lanczos filter
                        double weight = lanczos_2d(mR - m_in, nR - n_in, a);

                        // Add the weighted pixel value to the sum (iterate over channels)
                        for (int c = 0; c < image_in.channels(); ++c)
                        {
                            sum += weight * image_in(c, m_in, n_in);
                        }
                    }
                }
            }

            // Normalize the result, clamp the value, and assign to the output image
            for (int c = 0; c < image_out.channels(); ++c)
            {
                image_out(c, m, n) = clamp(static_cast<int>(sum), 0, image_out.range());
            }
        }
    }

    // Stop timer
    t = jbutil::gettime() - t;

    // Save output image
    std::ofstream file_out(outfile.c_str());
    image_out.save(file_out);

    // Show time taken
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
