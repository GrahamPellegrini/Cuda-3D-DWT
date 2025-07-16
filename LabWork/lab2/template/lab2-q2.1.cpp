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
    // If x is zero, return 1
    if (x == 0.0)
        return 1.0;
    // Otherwise, return the sinc function
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
    // Which is the same as the product of two 1D Lanczos filters
    return lanczos(x, a) * lanczos(y, a);
}


// Define a clamp function
double clamp(double x, double min, double max)
{
    // This function forces x to be within the range [min, max]
    // If x is less than min, return min
    if (x < min)
        return min;
    // If x is greater than max, return max
    if (x > max)
        return max;
    // Otherwise, return x
    return x;
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

    // Get input image parmaters
    int rows_in = image_in.get_rows();
    int cols_in = image_in.get_cols();
    int channels = image_in.channels();
    
    // Print the inputs image parameters
    std::cerr << "Input image: " << rows_in << " rows, " << cols_in << " columns, " << channels << " channels." << std::endl;

    // Start timer to measure time taken
    double t = jbutil::gettime();
    // This is the point where the output image starts to be calculated

    // Calculate output image dimensions based on scale factor
    int rows_out = static_cast<int>(rows_in * R);
    int cols_out = static_cast<int>(cols_in * R);


    // Print the output image parameters based on the scale factor
    std::cerr << "Resampling output image to " << rows_out << " rows, " << cols_out << " columns." << std::endl;

    // Create output image with 1 channel for greyscale
    jbutil::image<int> image_out(rows_out, cols_out, 1);

        
    // Pre compute the 2D Lanczos filter values
    std::vector<std::vector<double>> lanczos_weights(2 * a, std::vector<double>(2 * a));
    
    // Iterate over the 2D Lanczos filter values
    for (int i = -a + 1; i <= a; ++i)
    {
        for (int j = -a + 1; j <= a; ++j)
        {
            // Calculate the 2D Lanczos filter value and store it in the 2D vector. Note that we are storing the absolute value of the Lanczos filter value since the sinc function is symmetric.
            lanczos_weights[i + a - 1][j + a - 1] = abs(lanczos_2d(i, j, a));
        }
    }


    // Iterate over the output image pixels
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
                    int m_in = static_cast<int>(mR) + i;
                    int n_in = static_cast<int>(nR) + j;

                    // Make sure the neighboring pixel coordinates are within the input image bounds
                    if (m_in >= 0 && m_in < rows_in && n_in >= 0 && n_in < cols_in)
                    {
                        // Apply the Lanczos filter to the neighboring pixel to get the weight 
                        double weight = lanczos_weights[i + a - 1][j + a - 1];

                        // Add the weighted pixel value to the sum (considering greyscale image only)
                        sum += weight * image_in(0, m_in, n_in);
                        
                    }
                }
            }

            // Once the sum is calculated, clamp the value to the image range and assign it to the output image (greyscale image only)
            image_out(0, m, n) = clamp(static_cast<int>(sum), 0, image_out.range());
            
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

/*
The pre processing approach reduces the computation time by precomputing the 2D Lanczos filter values and storing them in a 2D vector. This way, the Lanczos filter values are computed only once and stored in the cache. Eliminating the standard repeation of calculated results.

However, this pre processing comes at the expense of memory usage. The 2D vector used to store the Lanczos filter values will consume memory space. The memory usage will increase with the size of the 2D vector, which is determined by the value of 'a'. The larger the value of 'a', the larger the 2D vector and the more memory space consumed.
*/