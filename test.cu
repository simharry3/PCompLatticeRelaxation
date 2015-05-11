////////////////////////////////////////////////////////////////////////////////
//                             CUDA LATTICE RELAXATION
//                           WRITTEN BY: CLAYTON RAYMENT
//
//    I wrote this program to help teach myself CUDA. While MATLAB and OCTAVE
//    are multithreaded applications, this program runs the majority of the
//    calculation directly on the GPU, which has the capability of being much
//    parallel than a standard CPU multithreaded application. Since I haven't
//    learned how to manage threads very well, the application is currently
//    running close to its memory limit. However, with a better implimentation
//    of thread handling, I would be able to utilize all 1664 CUDA cores on my 
//    GTX 970. Currently I only use blocks, which reduces that significantly.
//    The program is fully scalable, as the user can select how many divisions
//    each centimeter is split up into using the U global variable. Large values
//    of U however, exceed the maximum memory of the card due to my poor thread
//    management.
//
//////////////////////////////////////////////////////////////////////////////////
#include <time.h>
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
//===========
//GLOBALS:
//===========

//--------------
// VARIABLES
//--------------
const int U = 200; //Number of blocks per unit. This allows the grid spacing to be user defined.
const int H = 11*U; //Height of the outer box
const int W = 12*U; //Width of the outer box
const int h = 3*U; //Height of the inner box
const int w = 4*U; //Width of the inner box
const int N = H*W; //Number of threads required


//--------------
// FUNCTIONS
//--------------

//CUDA FUNCTIONS:

//Form the base simulation area:
__global__ void fill(double *a){
  //Form the outer rectangle:
  if(blockIdx.x < W || blockIdx.x > (N-W) || blockIdx.x % W == 0 || (blockIdx.x-(W-1))%W == 0){
    a[blockIdx.x] = 9;
  }
  else{
    a[blockIdx.x] = 1;
  }
  //Form the inner rectangle:
  if(blockIdx.x > W*((H-h)/2) && blockIdx.x < W*((H-h)/2+h) && blockIdx.x % W+1 > (W-w)/2 && blockIdx.x % W < (W-w)/2+w){
    a[blockIdx.x] = 0;
  }
}

//Perform one iteration of the matrix relaxation:
__global__ void average(double *a, double *b){
  //if we are at one of the edges, do nothing:
  if(a[blockIdx.x] == 0 || a[blockIdx.x] == 9){
    b[blockIdx.x] = a[blockIdx.x];
  }
  else{
    b[blockIdx.x] = (a[blockIdx.x]+a[blockIdx.x + 1]+a[blockIdx.x - 1]+a[blockIdx.x - W]+a[blockIdx.x + W])/5.0;
  }
}

//SERIAL FUNCTIONS:
void fillSerial(double *a){
    for(int i = 0; i < N; ++i){
        if(i < W || i > (N-W) || i%W == 0 || (i - (W-1))%W == 0){
            a[i] = 9;
        }
        else{
            a[i] = 4.5;
        }
        if( i > W*((H-h)/2) && i < W*((H-h)/2+h) && i % W+1 > (W-w)/2 && i % W < (W-w)/2+w){
            a[i] = 0;
        }
    }
}


void averageSerial(double* a, double* b){
    for(int i = 0; i < N; ++i){
        if(a[i] == 0 || a[i] == 9){
            b[i] = a[i];
        }
        else{
            b[i] = (a[i]+a[i+1]+a[i-1]+a[i+W]+a[i-W])/5.0;
        }
    }
}


//-----------------
//HELPER FUNCTIONS
//-----------------

//Quick function to print out the grid:
void print(double* a){
  for(int i = 0; i < N; ++i){
    if(i%W == 0){
      std::cout << '\n';
    }
    std::cout << a[i] << ' ';
  }
}


int main(void){
  //Serial Code:
    clock_t tS;
    clock_t tP;
    clock_t tT;
    clock_t tO;
    tS = clock();
    double* notPlate_s = new double[N]; //Serial copy of the non-plate locations
    double* tempVal_s = new double[N]; //Double array to hold temporary values
    for(int i = 0; i < 1000; ++i){
        fillSerial(notPlate_s);
        averageSerial(notPlate_s, tempVal_s);
        notPlate_s = tempVal_s;
    }
    tS = clock() - tS;


    tP = clock();
  //PARALLEL CODE:

  double *notPlate_h; //Host copy of the non-plate locations
  double *notPlate_d; //Device copy of the non-plate locations
  int size = N * sizeof(double);
  
  //The first thing we need to do is fill the initial experiment vector:
  cudaMalloc((void **)&notPlate_d, size); //Allocate memory on device for the not-plate vector
  notPlate_h = (double *)malloc(size); //Allocate memory on the host for the not-plate vector
  cudaMemcpy(notPlate_d, notPlate_h, size, cudaMemcpyHostToDevice);
  fill<<<N,1>>>(notPlate_d);
  cudaMemcpy(notPlate_h, notPlate_d, size, cudaMemcpyDeviceToHost);


  //Now we need to create a secondary vector to hold our rolling values while the GPU works:
  double *tempVal_h;
  double *tempVal_d;
  
  cudaMalloc((void **)&tempVal_d, size); //Allocate memory on the device for the temp vector
  tempVal_h = (double *)malloc(size); //allocate memory on the host for the temp vector
  
  
  //Since GPU time is cheap, we don't set a threshold, just run the simulation 1000 times:
  for(int i = 0; i < 1000; ++i){
    cudaMemcpy(notPlate_d, notPlate_h, size, cudaMemcpyHostToDevice);
    average<<<N,1>>>(notPlate_d, tempVal_d);
    cudaMemcpy(tempVal_h, tempVal_d, size, cudaMemcpyDeviceToHost);
    notPlate_h = tempVal_h;
  }
  //The output here will be redirected to a plaintext file which will then be opened and plot in OCTAVE
  //print(notPlate_h);

  //When uncommented, this prints to the std output what the values at the user input points are:  
 /* while(true){
    std::cout << "\nPlease enter an X coordinate: ";
    double x; 
    std::cin >> x;
    x = x*U;
    std::cout << "\nPlease enter a Y coordinate: ";
    double y;
    std::cin >> y;
    y = y*U;
    std::cout << "The point (" << x/U << "," << y/U << ") has value: " << notPlate_h[(int)(y*W+x)] << " Statvolts\n";
  }*/
  //Free the memory just to be safe:
  free(notPlate_h);
  cudaFree(notPlate_d);
  tP = clock() - tP;

  tO = clock();

  //OPENMP CODE
  #pragma omp parallel
  {
  double* notPlate_s = new double[N]; //Serial copy of the non-plate locations
  double* tempVal_s = new double[N]; //Double array to hold temporary values
  #pragma omp for
  for(int i = 0; i < 1000; ++i){
      fillSerial(notPlate_s);
      averageSerial(notPlate_s, tempVal_s);
      notPlate_s = tempVal_s;
  }
  }
  tO = clock() - tO;

  tT = clock();

  //THRUST CODE
  thrust::device_vector<double> notPlate_dT(N);
  thrust::host_vector<double> notPlate_hT(N);
  thrust::generate(notPlate_dT.begin(), notPlate_dT.end(),fillSerial());
  thrust::copy(notPlate_dT.begin(), notPlate_dR.end(), notPlate_hT.begin());

  thrust::host_vector<double> tempVal_hT(N);
  thrust::device_vector<double> tempVal_dT(N);

  for(int i = 0; i < 1000; ++i) {
    thrust::copy(notPlate_dT.begin(), notPlate_dT.end(), notPlate_hT.begin());
    thrust::transform(notPlate_dTbegin(), notPlate_dT.end(), tempVal_dT.begin(), tempVal_dT.begin(), averageSerial());
    thrust::copy(notPlate_hT.begin(), notPlate_hT.end(), notPlate_dT.begin());
    notPlate_hT = tempVal_hT;
  }

  tT = clock() - tT;
  std::cout << "N: " << N << " | SERIAL: " << (float)tS/CLOCKS_PER_SEC << "s | PARALLEL: " << (float)tP/CLOCKS_PER_SEC << "s | OPENMP: " << (float)tO/CLOCKS_PER_SEC << ""s | THRUST: ";
  std::cout << (float)tT/CLOCKS_PER_SEC << "s" << std::endl;
  return 0;
}
