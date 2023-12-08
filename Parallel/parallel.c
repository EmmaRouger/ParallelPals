#include <stdio.h>
#include <stdlib.h>
#include <png.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>

// #define WIDTH 100   // Define image width
// #define HEIGHT 100  // Define image height
#define CHANNELS 3  // Define the number of color channels

#define K 8  // Number of clusters for K-means
double start_time, end_time, start_time_kmeans; // for measuring time
double elapsed_time = 0;

// Structure to represent a pixel
typedef struct {
    unsigned char r, g, b;

} Pixel;

bool isPixelEqual(Pixel p1, Pixel p2)
{
    if((p1.r == p2.r) && (p1.g == p2.g) && (p1.b == p2.b))
        return true;
    return false;
}

// Calculate distance between two pixels
double calculateDistance(Pixel p1, Pixel p2) {
    return sqrt(pow(p1.r - p2.r, 2)+pow(p1.g - p2.g, 2)+ pow(p1.b - p2.b, 2));

}

//Update centroids based on assigned pixels
void updateCentroids(Pixel centroids[K], int clusterSizes[K], Pixel clusters[K])
{
    for (int i = 0; i < K; i++)
    {
        if (clusterSizes[i] > 0)
        {
            centroids[i].r = clusters[i].r / clusterSizes[i];
            centroids[i].g = clusters[i].g / clusterSizes[i];
            centroids[i].b = clusters[i].b / clusterSizes[i];
        }
    }
}

// K-means clustering on image pixels
Pixel** kMeans(Pixel centroids[K], Pixel **pixels, int width, int height, int numThreads) {

    int clusterSizes[K] = {0};
    int end=0;
    Pixel** clusteredPixels = (Pixel**)malloc(sizeof(Pixel*) * (height));

    
    for (int y = 0; y < height; y++) {
        clusteredPixels[y] = (Pixel*)malloc(sizeof(Pixel) * (width));//can cause issues - should allocate memory outside parallel region
        for (int x = 0; x < width; x++) {
            clusteredPixels[y][x].r = 0;
            clusteredPixels[y][x].g = 0;
            clusteredPixels[y][x].b = 0;
        }
    }
    while(end==0)
    {
        Pixel clusters[K] = {0};

        #pragma omp parallel for num_threads(numThreads)
        for (int i = 0; i < width; i++)
        {
            for (int j = 0; j < height; j++) {
                double minDistance = calculateDistance(centroids[0], pixels[i][j]);
                int closestCluster = 0;

                for (int k = 1; k < K; k++)
                {
                    double distance = calculateDistance(centroids[k], pixels[i][j]);
                    if (distance < minDistance)
                    {
                        minDistance = distance;
                        closestCluster = k;
                    }
                }

                #pragma critical
                {
                    clusters[closestCluster].r += pixels[i][j].r;
                    clusters[closestCluster].g += pixels[i][j].g;
                    clusters[closestCluster].b += pixels[i][j].b;
                    clusteredPixels[i][j].r = centroids[closestCluster].r;
                    clusteredPixels[i][j].g = centroids[closestCluster].g;
                    clusteredPixels[i][j].b = centroids[closestCluster].b;
                    clusterSizes[closestCluster]++;
                }
            }
        }

        Pixel oldCentroids[K];
        for(int i = 0; i < K; i++)
        {
            oldCentroids[i] = centroids[i];
        }
        updateCentroids(centroids, clusterSizes, clusters);
        for(int i=0; i < K; i++)
        {
            if(isPixelEqual(oldCentroids[i], centroids[i]))
            {
                end=1;
            }
            else
            {
                end=0;
                break;
            }
        }
        end=1;
    }

    return clusteredPixels;
}

// Function to read PNG file and create a 2D array of pixels
Pixel** readPNG(const char* filename, int* width, int* height) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return NULL;
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fclose(fp);
        fprintf(stderr, "Error initializing libpng\n");
        return NULL;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        fclose(fp);
        png_destroy_read_struct(&png, NULL, NULL);
        fprintf(stderr, "Error initializing PNG info\n");
        return NULL;
    }

    if (setjmp(png_jmpbuf(png))) {
        fclose(fp);
        png_destroy_read_struct(&png, &info, NULL);
        fprintf(stderr, "Error during PNG file reading\n");
        return NULL;
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    *width = png_get_image_width(png, info);
    *height = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    if (bit_depth == 16)
        png_set_strip_16(png);

    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);

    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);

    if (png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);

    if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    png_read_update_info(png, info);

    // Allocate memory for the row pointers
    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * (*height));
    for (int y = 0; y < *height; y++)
        row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png, info));

    png_read_image(png, row_pointers);

    fclose(fp);
    png_destroy_read_struct(&png, &info, NULL);

    // Create a 2D array of pixels
    Pixel** pixels = (Pixel**)malloc(sizeof(Pixel*) * (*height));
    for (int y = 0; y < *height; y++) {
        pixels[y] = (Pixel*)malloc(sizeof(Pixel) * (*width));
        for (int x = 0; x < *width; x++) {
            pixels[y][x].r = row_pointers[y][x * 4];
            pixels[y][x].g = row_pointers[y][x * 4 + 1];
            pixels[y][x].b = row_pointers[y][x * 4 + 2];
        }
    }

    // Free memory used for row pointers
    for (int y = 0; y < *height; y++)
        free(row_pointers[y]);
    free(row_pointers);

    return pixels;
}


// Function to free the memory used by the 2D array of pixels
void freePixels(Pixel** pixels, int height) {
    for (int y = 0; y < height; y++)
        free(pixels[y]);
    free(pixels);
}

// Function to write a 2D array of pixels to a PNG file
void writePNG(const char* filename, int width, int height, Pixel** pixels) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error opening file %s for writing\n", filename);
        return;
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fclose(fp);
        fprintf(stderr, "Error initializing libpng for writing\n");
        return;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        fclose(fp);
        png_destroy_write_struct(&png, NULL);
        fprintf(stderr, "Error initializing PNG info for writing\n");
        return;
    }

    if (setjmp(png_jmpbuf(png))) {
        fclose(fp);
        png_destroy_write_struct(&png, &info);
        fprintf(stderr, "Error during PNG file writing\n");
        return;
    }

    png_init_io(png, fp);

    // Set image information
    png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    // Allocate memory for row pointers
    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++)
        row_pointers[y] = (png_byte*)malloc(3 * width);

    // Copy pixel values to row pointers
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            row_pointers[y][x * 3] = pixels[y][x].r;
            row_pointers[y][x * 3 + 1] = pixels[y][x].g;
            row_pointers[y][x * 3 + 2] = pixels[y][x].b;
        }
    }

    // Write the image data
    png_write_image(png, row_pointers);
    png_write_end(png, NULL);

    // Free memory used for row pointers
    for (int y = 0; y < height; y++)
        free(row_pointers[y]);
    free(row_pointers);

    // Close the file
    fclose(fp);

    // Destroy the PNG structure
    png_destroy_write_struct(&png, &info);

}
int main(int argc, char*argv[])
{   
    start_time = MPI_Wtime();
    int rank, nproc, threads,height,width, work, start;
    const char *fileName;
    Pixel** localPixels;
    Pixel** pixels;
    Pixel* centroids;
    int *workArray;
    int *offset;

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(comm, &rank);

    if(rank == 0)
    {
        if(argc < 3)
        {
            printf("Usage: mpiexec -n <numOfProcs> <.exe> <inputFile> <numOfThreads>");
            MPI_Finalize();
            return -1;
        }

        threads = atoi(argv[2]);
        printf("%d\n", threads);

        fileName = argv[4];
        // Read the PNG file and get the 2D array of pixels
        pixels = readPNG(fileName, &width, &height);
        if (!pixels) {
            fprintf(stderr, "Error reading PNG file\n");
            return 1;
        }
        //calulate work and displacement for each process
        work = height/nproc;
        workArray = malloc(sizeof(int) * nproc);
        offset = malloc(sizeof(int)*nproc);
        for(int i = 0; i < nproc; i++)
        {
            workArray[i] = work;
            offset[i] = rank*work;
            if(rank == nproc-1)
            {
                workArray[i] = height-(rank*work);//read my git comment if you want to understand this right away
            }
        }
    }

    start_time_kmeans = MPI_Wtime();

    centroids = (Pixel*)malloc(sizeof(Pixel*) * (K));
    if(rank==0)
    {
        for (int i = 0; i < K; i++)
        {
            centroids[i].r = rand() % (255 - 0 + 1) + 0;
            centroids[i].g = rand() % (255 - 0 + 1) + 0;
            centroids[i].b = rand() % (255 - 0 + 1) + 0;
        }
    }
    
    MPI_Bcast(workArray,nproc,MPI_INT,0,comm);
    MPI_Bcast(offset, nproc, MPI_INT,0,comm);

    //brodcast the centroids
    MPI_Bcast(centroids, K, MPI_BYTE, 0, comm);

    // measure k-means time

    //all run kMeanks
    Pixel** clusteredImage = kmeans(centroids, localPixels, width, height, threads);
    end_time = MPI_Wtime();

    elapsed_time = end_time - start_time_kmeans;

    printf("Elapsed Time (K-Means): %ld\n", elapsed_time);
    //gather the pixels

    // if(rank == 0)
    // {
    //     writePNG("output.png", width, height, clusteredImage); // this will have to gather from all other process
    //     freePixels(pixels,height);
    // }

    freePixels(localPixels, height);
    free(centroids);
    MPI_Finalize();

    end_time = MPI_Wtime();

    elapsed_time = end_time - start_time;

    printf("Elapsed Time (Full): %ld\n", elapsed_time);

    return 0;
}