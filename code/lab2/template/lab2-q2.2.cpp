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
// Include the unordered_map library for the cache
#include <unordered_map>
// Include the utility library for the pair
#include <utility> 

// Define struct for cache hash
struct cache_hash
{
    // Define the hash function
    size_t operator()(const std::pair<int, int>& p) const
    {
        // Has the two elements of the pair
        auto h1 = std::hash<int>{}(p.first);
        auto h2 = std::hash<int>{}(p.second);
        // Combine the two hashes as a return result
        return h1 ^ h2;
    }
};

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

        
    // Using the defined cache hash struct
    std::unordered_map<std::pair<int, int>, double, cache_hash> cache;
    
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
                        // Create key for cache lookup based on scaled pixel positions
                        std::pair<int, int> cache_key = {i, j};

                        // Initialize weight to be set either from a previously computed value that was cached or one not yet computed and to be stored in the cache
                        double weight;

                        // Check if the Lanczos value is in the cache
                        auto it = cache.find(cache_key);
                        // If found in cache 
                        if (it != cache.end())
                        {
                            // Set the weight to the value found in the cache
                            weight = it->second; 
                        }
                        else
                        {
                            // If not found in cache, compute the Lanczos filter value
                            weight = lanczos_2d(i, j, a);

                            // Store the computed value in the cache
                            cache[cache_key] = weight;
                        }

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
The caching appraoch is implemented by using an unordered_map to store the computed Lanczos filter values. The key for the cache is a pair of integers representing the i and j values of the neighboring pixel. The cache is checked for the value of the Lanczos filter for the neighboring pixel. If the value is found in the cache, it is used directly. If the value is not found in the cache, the Lanczos filter value is computed and stored in the cache for future use. This approach reduces the number of redundant computations of the Lanczos filter values, improving the efficiency of the resampling process.

However, this appraoch is significantly slower than the pre-processing approach. This is because the cache lookup and insertion operations have a non-negligible overhead, especially when the cache size is large. The main expense of caching comes from cache misses which are inevitable and frequently occuring in this application.
*/