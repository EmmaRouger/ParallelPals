# ParallelPals
Image Segmentation with Parallel K-means

Authors: Tabitha Ristoff, Emma Rouger, Justin Turziak, Mitchell Misischia

Overview:
This project focuses on advancing image segmentation using parallel techniques. The goal is to develop an efficient and high-performance image segmentation algorithm using hybrid parallelization with MPI and OpenMP. The K-means algorithm groups pixels into clusters based on their proximity to centroids.

sequential.c
Compile: 
    1:  make
    2:  gcc -o out sequential.c -lpng
Run:
    ./out

parallel.c
Compile: 
    1: make
    2: mpicc -fopenmp -o out parallel.c -lpng
Run:
    mpiexec -n <numProcs> ./out <inputFile> <numThreads>

Brief descriptions of main functions
------------------------------------
Pixel** kMeans(Pixel centroids[K], Pixel **pixels, int width, int height, int numThreads):
Implements a clustering algorithm using openMP to achieve parallelism. It assigns pixels to clusters based on centroid distances, updating cluster information and centroids. For each pixel, it calculates the closest centroid using the distance function. To eliminate race conditions we used pragma omp critical to update the clusters RGB values. The function returns an array containing clustered pixel values.

void updateCentroids(Pixel centroids[K], int clusterSizes[K], Pixel clusters[K]):
Updates the centroids based on the accumulated pixel values and the number of pixels assigned to each cluster. It calculates the average RGB values of all
pixels assigned to each cluster.

Pixel** readPNG(const char* filename, int* width, int* height):
This function transfoms an image in png format into a 2D array of Pixel.

void writePNG(const char* filename, int width, int height, Pixel** pixels):
This function is responsible for opening up a file for writing, setting image information, allocating necessary memory, and writing the pixel data into the PNG file.






