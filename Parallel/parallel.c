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
Pixel* kMeans(Pixel centroids[K], Pixel *pixels, int width, int height, int numThreads) {

    //Keeps track of cluster sizes
    int clusterSizes[K] = {0};
    int end=0;

    //allocates memory for a 2D array to store clustered pixels
    Pixel* clusteredPixels = (Pixel*)malloc(sizeof(Pixel*) * (height)*width);
    omp_set_num_threads(numThreads);

    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = y * width + x;
            clusteredPixels[index].r = 0;
            clusteredPixels[index].g = 0;
            clusteredPixels[index].b = 0;
        }
    }
    while(end==0)
    {
        Pixel clusters[K] = {0};

        #pragma omp parallel for
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++) {
                int index = i * width + j;
                double minDistance = calculateDistance(centroids[0], pixels[index]);
                int closestCluster = 0;

                for (int k = 1; k < K; k++)
                {
                    double distance = calculateDistance(centroids[k], pixels[index]);
                    if (distance < minDistance)
                    {
                        minDistance = distance;
                        closestCluster = k;
                    }
                }

                #pragma omp critical
                {
                    clusters[closestCluster].r += pixels[index].r;
                    clusters[closestCluster].g += pixels[index].g;
                    clusters[closestCluster].b += pixels[index].b;
                }
                clusteredPixels[index].r = centroids[closestCluster].r;
                clusteredPixels[index].g = centroids[closestCluster].g;
                clusteredPixels[index].b = centroids[closestCluster].b;
                clusterSizes[closestCluster]++;
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
Pixel* readPNG(const char* filename, int* width, int* height) {
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

    // Create a 2D array of pixels -- doesnt work for scatter :(
    Pixel** twoDPixels = (Pixel**)malloc(sizeof(Pixel*) * (*height));
    for (int y = 0; y < *height; y++) {
        twoDPixels[y] = (Pixel*)malloc(sizeof(Pixel) * (*width));
        for (int x = 0; x < *width; x++) {
            twoDPixels[y][x].r = row_pointers[y][x * 4];
            twoDPixels[y][x].g = row_pointers[y][x * 4 + 1];
            twoDPixels[y][x].b = row_pointers[y][x * 4 + 2];
        }
    }
    //learned why scatter does not work on 2d arrays this is our last min fix
    Pixel* pixels= (Pixel*)malloc(sizeof(Pixel) * (*width* *height));
    int index = 0;
    for (int y = 0; y < *height; y++) {
        for (int x = 0; x < *width; x++) {
            pixels[index++] = twoDPixels[y][x];
        }
    }

    // Free memory used for row pointers
    for (int y = 0; y < *height; y++) {
        free(twoDPixels[y]);
    }
    free(twoDPixels);
    for (int y = 0; y < *height; y++)
        free(row_pointers[y]);
    free(row_pointers);

    return pixels;
}


// Function to write a 2D array of pixels to a PNG file
void writePNG(const char* filename, int width, int height, Pixel* pixels) {
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
            int index = y * (width) + x;
            row_pointers[y][x * 3] = pixels[index].r;
            row_pointers[y][x * 3 + 1] = pixels[index].g;
            row_pointers[y][x * 3 + 2] = pixels[index].b;
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
    int rank, nproc, threads,height,width, work, start;
    double startTime, endTime, startTimeKmeans, endTimeKmeans, elapsedTime; 

    const char *fileName;
    Pixel* localPixels;
    Pixel* pixels;
    Pixel *clusteredImage;
    Pixel* centroids;
    int *workArray;
    int *offset;

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(comm, &rank);
    MPI_Datatype pixel_type;
    MPI_Type_contiguous(3, MPI_UNSIGNED_CHAR, &pixel_type);
    MPI_Type_commit(&pixel_type);

    //Start the time for the whole program
    MPI_Barrier(comm);
    startTime = MPI_Wtime();
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

        fileName = argv[1];
        // Read the PNG file and get the 2D array of pixels
        pixels = readPNG(fileName, &width, &height);

        // MM || Remove
        int count = 0;
        for(int i = 0; i < width * height; i++)
            if(pixels[i].r ==0 || pixels[i].r)
                count++;
        printf("Count: %d\n",count);
        // MM || Remove END

        if (!pixels) {
            fprintf(stderr, "Error reading PNG file\n");
            return 1;
        }
        //calulate work and displacement for each process
        work = height/nproc;

        clusteredImage= (Pixel*)malloc(sizeof(Pixel) * (height)* width);
    }

    centroids = (Pixel*)malloc(sizeof(Pixel*) * (K));
    workArray = malloc(sizeof(int) * nproc);
    offset = malloc(sizeof(int)*nproc);

    if(rank==0)
    {
        for (int i = 0; i < K; i++)
        {
            centroids[i].r = rand() % (255 - 0 + 1) + 0;
            centroids[i].g = rand() % (255 - 0 + 1) + 0;
            centroids[i].b = rand() % (255 - 0 + 1) + 0;
        }
        for(int i = 0; i < nproc; i++)
        {
            workArray[i] = work;
            offset[i] = i*work;
            if(i == nproc-1)
            {
                workArray[i] = height-(i*work);//read my git comment if you want to understand this right away
            }
        }
    }

    MPI_Bcast(&width,1,MPI_INT,0,comm);
    MPI_Bcast(&threads,1,MPI_INT,0,comm);
    MPI_Bcast(workArray,nproc,MPI_INT,0,comm);
    MPI_Bcast(offset, nproc, MPI_INT,0,comm);
    work = workArray[rank];

    //brodcast the centroids
    MPI_Bcast(centroids, K, pixel_type, 0, comm);
    //allocate memory for local pixel
    localPixels = malloc(sizeof(Pixel) * work*width);
    // Scatter the pixel into the different
    MPI_Scatterv(pixels,workArray,offset,pixel_type, localPixels,workArray[rank],pixel_type,0,comm);

    //Start k-means time
    MPI_Barrier(comm);
    startTimeKmeans = MPI_Wtime();
    //all run kMeanks
    //Pixel* localClusteredImage= kMeans(centroids, localPixels, width, workArray[rank], threads);

    MPI_Barrier(comm);
    endTimeKmeans = MPI_Wtime();

    elapsedTime = endTimeKmeans - startTimeKmeans;
    printf("Elapsed Time (K-Means): %f\n", elapsedTime);

    //gather the pixels
    MPI_Gatherv(localPixels, workArray[rank], pixel_type, clusteredImage, workArray, offset, pixel_type, 0, comm);


    if(rank == 0)
    {
        writePNG("output.png", width, height, clusteredImage); // this will have to gather from all other process
        free(pixels);
    }

    //freePixels(localPixels, work);
    free(centroids);
    MPI_Barrier(comm);
    endTime = MPI_Wtime();

    elapsedTime = endTime - startTime;
    printf("Elapsed Time (Full): %f\n", elapsedTime);

    MPI_Finalize();

    return 0;
}