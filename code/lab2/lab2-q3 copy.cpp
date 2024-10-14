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
typedef int v4si __attribute__ ((vector_size (16)));

// Define the sinc function for SIMD
v4sf sinc(v4sf x)
{
    // Define constants
    v4sf vpi = {3.141592f, 3.141592f, 3.141592f, 3.141592f};
    v4sf zero = {0.0f, 0.0f, 0.0f, 0.0f};
    v4sf one = {1.0f, 1.0f, 1.0f, 1.0f};

    // If x is zero, return 1
    v4sf cmp0 = __builtin_ia32_cmpgeps(x, zero);

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


// Define a clamp function for SIMD
v4sf clamp(v4sf x, v4sf min, v4sf max)
{
    // This function forces x to be within the range [min, max]
    // If x is less than min, return min
    v4sf cmp_min = __builtin_ia32_cmpltps(x, min);
    x = __builtin_ia32_blendvps(cmp_min, min, x);

    // If x is greater than max, return max
    v4sf cmp_max = __builtin_ia32_cmpltps(max, x);
    x = __builtin_ia32_blendvps(cmp_max, max, x);

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

    // Get input image parameters
    int rows_in = image_in.get_rows();
    int cols_in = image_in.get_cols();
    int channels = image_in.channels();
    
    // Print the input image parameters
    std::cerr << "Input image: " << rows_in << " rows, " << cols_in << " columns, " << channels << " channels." << std::endl;

    // Start timer to measure time taken
    double t = jbutil::gettime();

    // Calculate output image dimensions based on the scale factor
    int rows_out = static_cast<int>(rows_in * R);
    int cols_out = static_cast<int>(cols_in * R);

    // Print the output image parameters based on the scale factor
    std::cerr << "Resampling output image to " << rows_out << " rows, " << cols_out << " columns." << std::endl;

    // Create output image with 1 channel for greyscale
    jbutil::image<int> image_out(rows_out, cols_out, 1);

    // Lanczos weight calculations for rows and columns
    v4sf r_weights = lanczos(a);
    v4sf c_weights = lanczos(a);

    // Iterate over the output image rows and columns, applying the Lanczos filter
    for (int m = 0; m < rows_out; m += 4)  // Process 4 rows at a time
    {
        for (int n = 0; n < cols_out; n += 4)  // Process 4 columns at a time
        {
            // Map output pixel (m, n) to input pixel space (float values)
            v4sf mR = {(m / R), (m + 1) / R, (m + 2) / R, (m + 3) / R};
            v4sf nR = {(n / R), (n + 1) / R, (n + 2) / R, (n + 3) / R};

            // Initialize sum for pixel contributions
            v4sf sum = {0.0f, 0.0f, 0.0f, 0.0f};

            // Perform Lanczos convolution within the window of size 'a'
            for (int i = -a + 1; i <= a; i += 1)
            {
                for (int j = -a + 1; j <= a; j += 1)
                {
                    // Create vectors for i and j
                    v4sf i_vec = {i, i + 1, i + 2, i + 3};
                    v4sf j_vec = {j, j + 1, j + 2, j + 3};

                    // Map input image coordinates and convert to integers
                    v4si m_in = __builtin_ia32_cvtps2dq(mR + i_vec);  // Convert float to int
                    v4si n_in = __builtin_ia32_cvtps2dq(nR + j_vec);  // Convert float to int

                    // Boundary checks: check if the coordinates are within the valid range
                    v4si valid_rows = __builtin_ia32_pand(
                        __builtin_ia32_pcmpgtd(m_in, v4si{0, 0, 0, 0}),  // m_in >= 0
                        __builtin_ia32_pcmpgtd(v4si{rows_in, rows_in, rows_in, rows_in}, m_in)  // m_in < rows_in
                    );
                    v4si valid_cols = __builtin_ia32_pand(
                        __builtin_ia32_pcmpgtd(n_in, v4si{0, 0, 0, 0}),  // n_in >= 0
                        __builtin_ia32_pcmpgtd(v4si{cols_in, cols_in, cols_in, cols_in}, n_in)  // n_in < cols_in
                    );
                    v4si valid_pixels = __builtin_ia32_pand(valid_rows, valid_cols);

                    // Calculate weight for each position using Lanczos
                    v4sf weight = r_weights[i + a - 1] * c_weights[j + a - 1];

                    // Load the pixel values from the input image
                    int idx1 = m_in[0] * cols_in + n_in[0];
                    int idx2 = m_in[1] * cols_in + n_in[1];
                    int idx3 = m_in[2] * cols_in + n_in[2];
                    int idx4 = m_in[3] * cols_in + n_in[3];

                    // Safely load pixel values, checking boundary conditions
                    v4sf pixel_values = {
                        static_cast<float>(valid_pixels[0] ? image_in(idx1) : 0),
                        static_cast<float>(valid_pixels[1] ? image_in(idx2) : 0),
                        static_cast<float>(valid_pixels[2] ? image_in(idx3) : 0),
                        static_cast<float>(valid_pixels[3] ? image_in(idx4) : 0)
                    };

                    // Multiply pixel values by the weight and sum
                    sum += pixel_values * weight;
                }
            }

            // Clamp the result to the valid image range [0, 255]
            sum = clamp(sum, v4sf{0.0f, 0.0f, 0.0f, 0.0f}, v4sf{255.0f, 255.0f, 255.0f, 255.0f});

            // Store the final result into the output image
            image_out(m, n) = static_cast<int>(sum[0]);
            image_out(m + 1, n) = static_cast<int>(sum[1]);
            image_out(m + 2, n) = static_cast<int>(sum[2]);
            image_out(m + 3, n) = static_cast<int>(sum[3]);
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