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


// Define the vectorized zero and one constants
const v4sf zero = {0.0f, 0.0f, 0.0f, 0.0f};
const v4sf one = {1.0f, 1.0f, 1.0f, 1.0f};
// Define the pi vectoirzed constant
const v4sf vpi = {3.141592f, 3.141592f, 3.141592f, 3.141592f};
// Define the epsilon constant for near-zero values
const v4sf epsilon = {1e-6, 1e-6, 1e-6, 1e-6};

// Vectorized clamp function for SIMD 
v4sf clamp(v4sf value, v4sf min_val, v4sf max_val) {
    // Apply min and max clamping
    value = __builtin_ia32_maxps(value, min_val); // Clamp to min
    value = __builtin_ia32_minps(value, max_val); // Clamp to max

    // Set values smaller than epsilon to zero
    v4sf near_zero_mask = __builtin_ia32_cmpltps(value, epsilon);
    // Set the values to zero where the value is less than epsilon
    value = __builtin_ia32_andnps(near_zero_mask, value); 
    return value;
}

// Define the SIMD sinc function
v4sf sinc(v4sf x) { 
    // Compare x with zero
    v4sf cmp0 = __builtin_ia32_cmpeqps(x, zero);

    // pix = π * x
    v4sf pix = x * vpi;
    
    // Initialize sinc_x with zeros
    v4sf sinc_x = zero;

    // Calculate sin(πx) for non-zero values using scalar sinf() on individual elements
    sinc_x[0] = sinf(pix[0]);
    sinc_x[1] = sinf(pix[1]);
    sinc_x[2] = sinf(pix[2]);
    sinc_x[3] = sinf(pix[3]);

    // Avoid division by zero: if x is zero, use one, otherwise perform sin(πx)/(πx)
    sinc_x = __builtin_ia32_andnps(cmp0, sinc_x / pix) + __builtin_ia32_andps(cmp0, one);

    return sinc_x; // Return the result
}

// Unroll the lancoz function to vecotrize the weights calculation
std::vector<float> lanczos(int a)
{   
    // Initialize the weights vector
    std::vector<float> weights;
    // Define constants for zero, one and a in vector form
    v4sf va = {a, a, a, a};

    // Loop unrolling for the weights calculation
    for (float i = -a + 1; i <= a; i+=4)
    {
        v4sf x = {i, i + 1, i + 2, i + 3};  // Vector of x values

        // Apply sinc(x) 
        v4sf sinc_x = sinc(x); 
        // Calculate x/va
        v4sf x_div_a = x/va;         
        // Apply sinc(x/a) 
        v4sf sinc_x_div_a = sinc(x_div_a); 

        // Calculate the weights: sinc(x) * sinc(x / a)
        v4sf w = sinc_x * sinc_x_div_a;

        // Take the absolute value of the weights by ANDing the weights with itself
        v4sf abs_w = __builtin_ia32_andps(w, w);

        // Clamp the weights to zero if the weights are less than epsilon
        abs_w = clamp(abs_w, zero, one);

        // Loop through the weights and store them in the weights vector
        for (int j = 0; j < 4; j++)
        {
            weights.push_back(abs_w[j]);
        }
    }
    // Return the weights vector
    return weights;
}

// Iteration over lanczos window to calculate the sum
float window(const jbutil::image<int> &image_in, const std::vector<float> &r_weights, const std::vector<float> &c_weights, const int m, const int n, const int a, const float R)
{
    // Initialize sum for pixel contributions
    float sum = 0.0f;

    // Iterate over the Lanczos window
    for (int i = -a + 1; i <= a; ++i)
    {
        for (int j = -a + 1; j <= a; ++j)
        {
            // Map the input pixel (m, n) to the output pixel space (float values)
            float mR = m / R;
            float nR = n / R;

            // Calculate the input pixel coordinates
            float m_in = mR + i;
            float n_in = nR + j;

            // Calculate the pixel value from the input image
            int pixel = image_in(0, m_in, n_in);

            // Calculate the weight index
            int w_index = (i + a - 1) * (2 * a - 1) + (j + a - 1);

            // Calculate the pixel contribution
            sum += pixel * r_weights[w_index] * c_weights[w_index];
        }
    }

    return sum;
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
    std::vector<float> r_weights = lanczos(a);
    std::vector<float> c_weights = lanczos(a);
    
     // Maximum value for the image
    float range = image_out.range();
    v4sf max_range = {range, range, range, range};
    
    // Iterate over the output image rows and columns, applying the Lanczos filter
    for (int m = 0; m < rows_out; m += 4)  // Process 4 rows at a time
    {
        for (int n = 0; n < cols_out; n += 4)  // Process 4 columns at a time
        {   

            // Boundary workaround for the output image
            int m_start = std::max(0, static_cast<int>(ceil(m / R - a)));
            int m_end = std::min(cols_in - 1, static_cast<int>(floor(m / R + a)));
            int n_start = std::max(0, static_cast<int>(ceil(n / R - a)));
            int n_end = std::min(rows_in - 1, static_cast<int>(floor(n / R + a)));

            // Initialize sum for pixel contributions
            v4sf sum = zero;

            // Iterate over the Lanczos window
            for (float i = m_start; i <= m_end; ++i)
            {
                for (float j = n_start; j <= n_end; ++j)
                {
                    
                    // Calculate the input pixel coordinates using the ceiling to get the nearest integer for pixel indexing
                    v4sf m_in = {ceil(m / R + i), ceil(m / R + i + 1), ceil(m / R + i + 2), ceil(m / R + i + 3)};
                    v4sf n_in = {ceil(n / R + j), ceil(n / R + j + 1), ceil(n / R + j + 2), ceil(n / R + j + 3)};

                    // Take the absolute value of the input pixel coordinates
                    m_in = __builtin_ia32_andps(m_in, m_in);
                    n_in = __builtin_ia32_andps(n_in, n_in);

                    // Calculate the pixel value from the input image
                    v4sf vec_pixel =
                     {static_cast<float>(image_in(0, m_in[0], n_in[0])), static_cast<float>(image_in(0, m_in[1], n_in[1])), static_cast<float>(image_in(0, m_in[2], n_in[2])),static_cast<float>(image_in(0, m_in[3], n_in[3]))};
                    
                    // Calculate the weight 
                    float weight = r_weights[(i + a - 1)] * c_weights[(j + a - 1)];

                    // Vectorize the weight
                    v4sf vec_weight = {weight, weight, weight, weight};

                    // Calculate the sum by multiplying the pixel value with the weight
                    sum += vec_pixel * vec_weight;
                }
            }
            // Clamp the sum to the maximum range
            sum = clamp(sum, zero, max_range);

            //  Unroll the vectorized sum to store the pixel values in the output image
            image_out(0, m, n) = sum[0];
            image_out(0, m + 1, n) = sum[1];
            image_out(0,m + 2, n) = sum[2];
            image_out(0, m + 3, n) = sum[3];
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

