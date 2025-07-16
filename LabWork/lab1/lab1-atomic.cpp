#include "../shared/jbutil.h"
#include <thread>
#include <atomic>

// Define the integration limits
#define a -2
#define b 2

// Define the function limits 
#define A 0
#define B 0.4

// Define mean and variance
#define mean 0
#define variance 1

// Define number of threads
#define num_threads 4

// Declaring the Given distribution function f outside the MonteCarloThread function
auto f = [](double x) -> double {
    return (1.0 / sqrt(2.0 * pi * variance * variance)) * exp(-pow(x - mean, 2) / (2.0 * variance * variance));
};

// Threaded Monte Carlo integration function
void MonteCarloThread(const int N, std::atomic<int>& total_count)
{
    // Declare random number generator
    jbutil::randgen rng;
    // Seed the random number generator depending on the thread id
    rng.seed(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    // Initialise count for samples within each thread
    int local_count = 0;

    // Iterate over samples
    for (int i = 0; i < N; ++i)
    {
        // Generate random numbers for x_i and y_i
        double x = rng.fval(a, b);
        double y = rng.fval(A, B);

        // Calculate F(x_i)
        double f_x = f(x);

        // Check if y_i is below the function
        if (y < f_x)
        {
            // If so, increment the count
            ++local_count;
        }
    }

    // Print the local count to check that each thread is individually counting
    std::cerr << "Local count: " << local_count << std::endl;

    // Add the local count to the total count
    total_count += local_count;
}

// Multithreaded Monte Carlo integration function
void MonteCarlo(const int N)
{
    std::cerr << "\nMultithreaded Implementation (" << N << " samples)" << std::endl;

    // Start timer
    double t = jbutil::gettime();

    // Determine number of samples per thread
    int samples_per_thread = N / num_threads;

    // Create a vector to hold the thread
    jbutil::vector<std::thread, 128> threads;
    // Create an atomic int to hold the total count
    std::atomic<int> total_count(0);


    // Launch threads
    for (int i = 0; i < num_threads; ++i)
    {
        // Create a thread for each thread usign the MonteCarloThread function and passing the divided samples and the count as reference (safe to pass by reference as each thread will write to a different location)
        threads.push_back(std::thread(MonteCarloThread, samples_per_thread,std::ref(total_count)));
    }

    // Join threads to wait for them to finish
    for (auto &t : threads)
    {
        t.join();
    }

    // Integral estimate
    double integral_estimate = (total_count / static_cast<double>(N)) * (b - a) * (B - A);

    // Stop timer
    t = jbutil::gettime() - t;

    // Error function for integral estimate (using predefined erf() function)
    double error = std::erf(std::sqrt(2));

    // Print results
    std::cerr << "Samples: " << N << std::endl;
    std::cerr << "Samples below the function: " << total_count << std::endl;
    std::cerr << "--------------------------------" << std::endl;
    std::cerr << "Integral Estimate: " << integral_estimate << std::endl;
    std::cerr << "Error: " << error << std::endl;
    std::cerr << "Time taken: " << t << "s" << std::endl;
}

int main()
{
    std::cerr << "Lab 1 Question 2: Multithreaded Monte Carlo integration" << std::endl;

    // Define number of samples
    const int N = int(1E8); 

    // Perform Monte Carlo integration using multithreading
    MonteCarlo(N);

    return 0;
}

/*
Q2. Update your implementation to make use of the four cores in the main   CPU, using threads.
Issues to consider:
a)How do you divide the problem?
b) Are all the functions used within the threads re‐entrant? (i.e. are they thread‐safe?)
c) Find an alternative for non‐thread‐safe functions.
-------------------------------------------------------------------------------
a) Identifying the divisable variable in the problem, that is the number of samples. We can divide the number of samples by the number of threads to get the number of samples per thread. This way each thread can work independently on a subset of the samples, at the same time. A struct system can be used to add the results of each thread together in a shared variable. Giving us the final result, in a divided manner.

b&c) The use of the jbuilt imports for vectors,randgen and gettime() are all thread-safe, the function f is also thread-safe as it does not use any shared variables. The logic of the error fucntion being thread safe or not is not important as it is not used in the threads. The use of an atomic total count ensures that the threads can write to the same location in memory without any issues. Therefore, after all these thread safetly checks and work arounds, the implementation is thread-safe.
*/
