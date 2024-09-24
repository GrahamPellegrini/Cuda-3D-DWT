
#include <iostream>
#include <cassert>

__global__ void VecAdd(int n, float *c, const float *a, const float *b)
   {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if(i < n)
      c[i] = a[i] + b[i];
   }

void test1d()
   {
   const int n = 1024;
   float c[n], a[n], b[n];

   // Fill in inputs
   for(int i = 0; i < n; i++)
      {
      a[i] = i;
      b[i] = 1024-i;
      }

   // Allocate arrays in device memory
   const int size = n * sizeof(float);
   float *dc, *da, *db;
   cudaMalloc((void **)&da, size);
   cudaMalloc((void **)&db, size);
   cudaMalloc((void **)&dc, size);

   // Copy over input from host to device
   cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);
   cudaMemcpy(db, b, size, cudaMemcpyHostToDevice);

   const int nblocks = (n + 63) / 64;
   VecAdd<<<nblocks, 64>>>(n, dc, da, db);

   // Copy over output from device to host
   cudaMemcpy(c, dc, size, cudaMemcpyDeviceToHost);

   // Free device memory
   cudaFree(da);
   cudaFree(db);
   cudaFree(dc);

   // Show results
   for(int i = 0; i < n; i++)
      {
      std::cout << i << '\t' << a[i] << '\t' << b[i] << '\t' << c[i] << '\n';
      assert(c[i] == a[i] + b[i]);
      }
   }

__global__ void MatAdd(int n, float *c, const float *a, const float *b, int pitch)
   {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   int j = blockIdx.y * blockDim.y + threadIdx.y;
   int ij = i * pitch + j;
   if(i < n && j < n)
      c[ij] = a[ij] + b[ij];
   }
   
void test2d()
   {
   const int n = 32;
   float c[n][n], a[n][n], b[n][n];

   // Fill in inputs
   for(int i = 0; i < n; i++)
      for(int j = 0; j < n; j++)
         {
         a[i][j] = i+j;
         b[i][j] = n*n-i-j;
         }

   // Allocate arrays in device memory
   const int rowsize = n * sizeof(float);
   float *dc, *da, *db;
   size_t pitch;
   cudaMallocPitch((void**)&da, &pitch, rowsize, n);
   cudaMallocPitch((void**)&db, &pitch, rowsize, n);
   cudaMallocPitch((void**)&dc, &pitch, rowsize, n);
   
   // Copy over input from host to device
   cudaMemcpy2D(da, pitch, a, rowsize, rowsize, n, cudaMemcpyHostToDevice);
   cudaMemcpy2D(db, pitch, b, rowsize, rowsize, n, cudaMemcpyHostToDevice);
   
   dim3 blocksize(16, 16);
   dim3 gridsize( (n + blocksize.x - 1) / blocksize.x,
                  (n + blocksize.y - 1) / blocksize.y);
   assert(pitch % sizeof(float) == 0);
   MatAdd<<<gridsize, blocksize>>>(n, dc, da, db, pitch / sizeof(float));

   // Copy over output from device to host
   cudaMemcpy2D(c, rowsize, dc, pitch, rowsize, n, cudaMemcpyDeviceToHost);

   // Free device memory
   cudaFree(da);
   cudaFree(db);
   cudaFree(dc);

   // Show results
   for(int i = 0; i < n; i++)
      for(int j = 0; j < n; j++)
         {
         std::cout << c[i][j] << ((j == n-1) ? '\n' : '\t');
         assert(c[i][j] == a[i][j] + b[i][j]);
         }
   }

int main()
   {
   test1d();
   test2d();
   
   return 0;
   }
