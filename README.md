# ParallelPals
k-means parallel implementation
Image Segmentation with Parallel K-means

Authors: Tabitha, Emma, Justin, Mitchell
Authors: Tabitha Ristoff, Emma Rouger, Justin Turziak, Mitchell Misischia

Overview:
This project focuses on advancing image segmentation using parallel techniques. The goal is to develop an efficient and high-performance image segmentation algorithm using hybrid parallelization with MPI and OpenMP. The K-means algorithm groups pixels into clusters based on their proximity to centroids.

sequential.c
Compile: make
Usage: 

parallel.c
Compile: 
    1: make
    2: mpicc -o 
Run:

Usage: mpiexec -n <numOfProcs> ./image_segmentation <inputFile.png> <numOfThreads>