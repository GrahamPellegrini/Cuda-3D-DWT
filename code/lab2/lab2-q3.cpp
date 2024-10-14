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
// Include for SIMD Vecotrization
#include <immintrin.h>

// Define the SIMD data type for 4 integers
typedef float v4sf __attribute__ ((vector_size (16)));

// Define the sinc function for SIMD
v4sf sinc(v4sf x)
{
    // Define constants
    v4sf vpi = {3.141592f, 3.141592f, 3.141592f, 3.141592f};
    v4sf zero = {0.0f, 0.0f, 0.0f, 0.0f};
    v4sf one = {1.0f, 1.0f, 1.0f, 1.0f};

    // If x is zero, return 1
    v4sf cmp0 = __builtin_ia32_cmpeqps(x, zero);

    // Compute sin(πx) using a simple approximation 
    v4sf pix = vpi * x;
    // Taylor series approximation
    v4sf sin_pix = pix - (pix * pix * pix) / 6.0f + (pix * pix * pix * pix * pix) / 120.0f;  

    // sinc(x) = sin(πx) / (πx)
    v4sf sinc_x = sin_pix / pix;

    // Use blending: if x == 0, return 1, otherwise return sinc(x)
    sinc_x = __builtin_ia32_blendvps(cmp0, one, sinc_x);
}

// Unroll the lancoz function to vecotrize the weights calculation
v4sf lanczos(int a)
{   
    // Initialize the weights vector
    v4sf weights;
    // Define constants for zero, one and a in vector form
    v4sf zero = {0.0f, 0.0f, 0.0f, 0.0f};
    v4sf one = {1.0f, 1.0f, 1.0f, 1.0f};
    v4sf va = {(float)a, (float)a, (float)a, (float)a};  // Vector for `a`

    // Loop unrolling for the weights calculation
    for (int i = -a + 1; i <= a; i+=4)
    {
         v4sf x = {i, i + 1, i + 2, i + 3};  // Vector of x values

        // Apply sinc(x) and sinc(x/a)
        v4sf sinc_x = sinc(x);             // sinc(x)
        v4sf sinc_x_div_a = sinc(x / va);  // sinc(x / a)

        // Calculate the weights: sinc(x) * sinc(x / a)
        v4sf w = sinc_x * sinc_x_div_a;
        // Add the weights to the weights vector
        weights += w;
    }
    // Return the weights vector
    return weights;
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

        
    // Get the r_weights and c_weights from the adjusted vectorized lanczos function
    v4sf r_weights = lanczos(a);
    v4sf c_weights = lanczos(a);


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
                        // Apply the precomputed Lanczos filter weights to get the 2D weights 
                        double weight = r_weights[i + a - 1] * c_weights[j + a - 1];

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

*/