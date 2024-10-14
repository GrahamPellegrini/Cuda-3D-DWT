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
#include <emmintrin.h>  // SSE2 intrinsics

// Define the SIMD data type for 4 integers
typedef float v4sf __attribute__ ((vector_size (16)));
typedef int v4si __attribute__ ((vector_size (16)));

// Define the vectorized zero and one constants
const v4sf zero = {0.0f, 0.0f, 0.0f, 0.0f};
const v4sf one = {1.0f, 1.0f, 1.0f, 1.0f};
// Define the pi vectoirzed constant
const v4sf vpi = {3.141592f, 3.141592f, 3.141592f, 3.141592f};

// Define the sinc function for SIMD
v4sf sinc(v4sf x)
{   
    // Absolute value of x can be calculated using the AND operation
    v4sf abs_x = __builtin_ia32_andps(x,x);

    // Compare x equal to zero
    v4sf cmp0 = __builtin_ia32_cmpeqps(x, zero);

    // pix = πx
    v4sf pix = {vpi[0] * x[0], vpi[1] * x[1], vpi[2] * x[2], vpi[3] * x[3]};
    
    // Initialize sinc_x with zeros
    v4sf sinc_x = {0.0f, 0.0f, 0.0f, 0.0f};

    // Calculate sin(πx) for non-zero values
    v4sf sin_pix = {sin(pix[0]), sin(pix[1]), sin(pix[2]), sin(pix[3])};

    // Only perform division where pix is not zero
    v4sf non_zero_pix = __builtin_ia32_andnps(cmp0, pix); // Get pix where x is not zero
    sinc_x = sin_pix / non_zero_pix; // Division by non-zero pix

    // Set sinc(x) to 1 where x is zero
    sinc_x = __builtin_ia32_andps(cmp0, one) + __builtin_ia32_andnps(cmp0, sinc_x);

    // Return the sinc(x) vector
    return sinc_x;
}

// Unroll the lancoz function to vecotrize the weights calculation
std::vector<double> lanczos(int a)
{   
    // Initialize the weights vector
    std::vector<double> weights;
    // Define constants for zero, one and a in vector form
    v4sf va = {a, a, a, a};

    // Loop unrolling for the weights calculation
    for (int i = -a + 1; i <= a; i+=4)
    {
        v4sf x = {i, i + 1, i + 2, i + 3};  // Vector of x values

        // Apply sinc(x) 
        v4sf sinc_x = sinc(x); 
        // Calculate x/va
        v4sf x_div_a = __builtin_ia32_divps(x, va);         
        // Apply sinc(x/a) 
        v4sf sinc_x_div_a = sinc(x_div_a); 

        // Calculate the weights: sinc(x) * sinc(x / a)
        v4sf w = sinc_x * sinc_x_div_a;

        // Take the absolute value of the weights using an abs operation
        v4sf abs_w = {abs(w[0]), abs(w[1]), abs(w[2]), abs(w[3])};

        // Store the weights in the vector
        weights.push_back(w[0]);
        weights.push_back(w[1]);
        weights.push_back(w[2]);
        weights.push_back(w[3]);
        
    }
    // Return the weights vector
    return weights;
}

// Define a clamp function for SIMD
v4sf clamp(v4sf x, v4sf min, v4sf max) {
    // Create flags for comparisons
    v4sf cmp_min = __builtin_ia32_cmpltps(x, min); // True if x < min
    v4sf cmp_max = __builtin_ia32_cmpltps(max, x); // True if x > max

    // Create minimum value where x is below min
    v4sf mini = __builtin_ia32_andps(cmp_min, min);

    // Create maximum value where x is above max
    v4sf maxi = __builtin_ia32_andps(cmp_max, max);

    // Retain x where it is within the bounds (not below min or above max)
    x = __builtin_ia32_andnps(cmp_min, x); // x remains unchanged if x >= min
    x = __builtin_ia32_andnps(cmp_max, x); // x remains unchanged if x <= max

    // Combine the results to get the final clamped value
    x = __builtin_ia32_orps(x, mini); // Incorporate min where applicable
    x = __builtin_ia32_orps(x, maxi); // Incorporate max where applicable
    
    // Return the clamped vector
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

    // Lanczos weight calculations for rows and columns stored in vectors
    std::vector<double> r_weights = lanczos(a);
    std::vector<double> c_weights = lanczos(a);

    // Iterate over the output image rows and columns, applying the Lanczos filter
    for (int m = 0; m < rows_out; m += 4)  // Process 4 rows at a time
    {
        for (int n = 0; n < cols_out; n += 4)  // Process 4 columns at a time
        {
            // Map output pixel (m, n) to input pixel space (float values)
            v4sf mR = {(m / R), (m + 1) / R, (m + 2) / R, (m + 3) / R};
            v4sf nR= {(n / R), (n + 1) / R, (n + 2) / R, (n + 3) / R};

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
                    v4sf m_in = mR + i_vec;
                    v4sf n_in = nR + j_vec;

                    // Define the max row and column values for the input image
                    v4sf rows_in_max = {rows_in - 1, rows_in - 1, rows_in - 1, rows_in - 1};
                    v4sf cols_in_max = {cols_in - 1, cols_in - 1, cols_in - 1, cols_in - 1};

                    // Clamp the input image coordinates to the image boundaries
                    m_in = clamp(m_in, zero, rows_in_max);
                    n_in = clamp(n_in, zero, cols_in_max);

                    // Vector for the rows in and coloumns in
                    v4sf rv = {rows_in - 1, rows_in - 1, rows_in - 1, rows_in - 1};
                    v4sf cv = {cols_in - 1, cols_in - 1, cols_in - 1, cols_in - 1};

                    // Check if the input pixel is within the image boundaries with 4 flags
                    v4sf m_in_zero = __builtin_ia32_cmpltps(m_in, zero);
                    // True if m_in < 0
                    v4sf m_in_rows = __builtin_ia32_cmpltps(rv, m_in);
                    // True if m_in > rows_in
                    v4sf n_in_zero = __builtin_ia32_cmpltps(n_in, zero);
                    // True if n_in < 0
                    v4sf n_in_cols = __builtin_ia32_cmpltps(cv, n_in);
                    // True if n_in > cols_in

                    // Combine the m_in invalid flags
                    v4sf m_in_invalid = __builtin_ia32_andps(m_in_zero, m_in_rows);

                    // Combine the n_in invalid flags
                    v4sf n_in_invalid = __builtin_ia32_andps(n_in_zero, n_in_cols);
            
                    // Combine the flags m_in_invalid and n_in_invalid to get the invalid flag
                    v4sf invalid = __builtin_ia32_andps(m_in_invalid, n_in_invalid);
            
                    // Calculate weight for each position using Lanczos
                    v4sf weight = {r_weights[i + a - 1] * c_weights[j + a - 1], r_weights[i + a] * c_weights[j + a], r_weights[i + a + 1] * c_weights[j + a + 1], r_weights[i + a + 2] * c_weights[j + a + 2]};

                    // Load the pixel values corresponding to the weights from the input image into vectorized form 
                    v4sf vec_pixel = {
                       image_in(0,m_in[0],n_in[0]),
                       image_in(0,m_in[1],n_in[1]),
                       image_in(0,m_in[2],n_in[2]),
                       image_in(0,m_in[3],n_in[3])
                    };

                    // Multiply pixels by the weights in a vectorized form
                    v4sf pixel_values = vec_pixel * weight;

                    // Take the absolute value of the pixel values by ANDing the pixel values with itself
                    pixel_values = __builtin_ia32_andps(pixel_values, pixel_values);
                
                    // Add the pixel values to the sum, using the invalid flag to set the pixel values to zero
                    sum += __builtin_ia32_andnps(invalid, pixel_values);
                }
            }

            // Maximum value for the image
            v4sf max_range = {image_out.range(), image_out.range(), image_out.range(), image_out.range()};

            // Clamp the sum to the image range
            sum = clamp(sum, zero, max_range);

            // Store the final result into the output image
            image_out(0, m, n) = static_cast<int>(sum[0]);
            image_out(0, m + 1, n) = static_cast<int>(sum[1]);
            image_out(0, m + 2, n) = static_cast<int>(sum[2]);
            image_out(0, m + 3, n) = static_cast<int>(sum[3]);
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