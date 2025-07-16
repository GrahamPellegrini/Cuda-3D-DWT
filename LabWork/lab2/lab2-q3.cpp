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
    
    // Initialize sin_x to the sin of pix
    v4sf sin_x = {sin(pix[0]), sin(pix[1]), sin(pix[2]), sin(pix[3])};

    // Calculate the sinc function: sin(πx)/(πx)
    v4sf sinc_x = sin_x / pix;
    // Avoid division by zero: if x is zero, use one, otherwise perform sin(πx)/(πx)
    sinc_x = __builtin_ia32_andnps(cmp0, sinc_x) + __builtin_ia32_andps(cmp0, one);

    return sinc_x; // Return the result
}

// Unroll the lancoz function to vecotrize the weights calculation
std::vector<double> lanczos(int a)
{   
    // Initialize the weights vector
    std::vector<double> weights;
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
            // Map output pixel (m, n) to input pixel space (float values), taking the ceiling to bring int values for indexing
            v4sf mR = {ceil((m/R)), ceil((m + 1)/R), ceil((m + 2)/R), ceil((m + 3)/R)};
            v4sf nR = {ceil((n/R)), ceil((n + 1)/R), ceil((n + 2)/R), ceil((n + 3)/R)};

            // Initialize sum for pixel contributions
            v4sf sum = zero;

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
                    v4sf weight = {r_weights[i + a -1 ] * c_weights[j + a - 1], r_weights[i + a] * c_weights[j + a], r_weights[i + a + 1] * c_weights[j + a + 1], r_weights[i + a + 2] * c_weights[j + a + 2]};

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

            //std::cerr << "Sum before clamp: " << sum[0] << " " << sum[1] << " " << sum[2] << " " << sum[3] << std::endl;
            sum = clamp(sum, zero, max_range);
            //std::cerr << "Sum after clamp: " << sum[0] << " " << sum[1] << " " << sum[2] << " " << sum[3] << std::endl;



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