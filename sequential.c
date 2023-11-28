#include <stdio.h>
#include <stdlib.h>
#include <png.h>
#include <stdbool.h>

#define WIDTH 100   // Define image width
#define HEIGHT 100  // Define image height
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
    return sqrt(pow(p1.r - p2.r, 2) + pow(p1.g - p2.g, 2) + pow(p1.b - p2.b, 2));
}

// Update centroids based on assigned pixels
void updateCentroids(Pixel centroids[K], int clusterSizes[K], Pixel pixels[WIDTH][HEIGHT]) {
    for (int i = 0; i < K; i++) {
        if (clusterSizes[i] > 0) {
            centroids[i].r /= clusterSizes[i];
            centroids[i].g /= clusterSizes[i];
            centroids[i].b /= clusterSizes[i];
        }
    }
}

// K-means clustering on image pixels
void kMeans(Pixel centroids[K], Pixel pixels[WIDTH][HEIGHT]) {
    int clusterSizes[K] = {0};
    int end=0;
    while(end==0)
    {
        for (int i = 0; i < WIDTH; i++)
        {
            for (int j = 0; j < HEIGHT; j++) {
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

                centroids[closestCluster].r += pixels[i][j].r;
                centroids[closestCluster].g += pixels[i][j].g;
                centroids[closestCluster].b += pixels[i][j].b;
                clusterSizes[closestCluster]++;
            }
        }

        Pixel oldCentroids[K];
        for(int i = 0; i < K; i++)
        {
            oldCentroids[i] = centroids[i];
        }
        updateCentroids(centroids, clusterSizes, pixels);
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
    }
}

int main() {
    Pixel pixels[WIDTH][HEIGHT];
    Pixel centroids[K];

    // Initialize centroids and pixels, read from a PNG image using libpng

    FILE *file = fopen("input.png", "rb");
    if (!file) {
        printf("Failed to open the image file\n");
        return -1;
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fclose(file);
        printf("Failed to create PNG read structure\n");
        return -1;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, (png_infopp)NULL, (png_infopp)NULL);
        fclose(file);
        printf("Failed to create PNG info structure\n");
        return -1;
    }

    png_init_io(png, file);
    png_read_info(png, info);

    if (png_get_color_type(png, info) != PNG_COLOR_TYPE_RGB) {
        printf("The input image is not in RGB format\n");
        return -1;
    }

    if (png_get_bit_depth(png, info) != 8) {
        printf("The input image does not have 8-bit depth\n");
        return -1;
    }

    if (png_get_image_width(png, info) != WIDTH || png_get_image_height(png, info) != HEIGHT) {
        printf("Image size does not match specified dimensions\n");
        return -1;
    }

    for (int i = 0; i < K; i++) {
        centroids[i] = pixels[rand() % WIDTH][rand() % HEIGHT];
    }

    png_bytep row;
    for (int i = 0; i < HEIGHT; i++) {
        row = (png_bytep)&pixels[i];
        png_read_row(png, row, NULL);
    }

    fclose(file);
    png_destroy_read_struct(&png, &info, NULL);

    // Perform K-means clustering
    kMeans(centroids, pixels);

    // Save the clustered image (output) using libpng

    FILE *outputFile = fopen("output.png", "wb");
    if (!outputFile) {
        printf("Failed to create the output file\n");
        return -1;
    }

    png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fclose(outputFile);
        printf("Failed to create PNG write structure\n");
        return -1;
    }

    info = png_create_info_struct(png);
    if (!info) {
        fclose(outputFile);
        png_destroy_write_struct(&png, (png_infopp)NULL);
        printf("Failed to create PNG info structure\n");
        return -1;
    }

    png_init_io(png, outputFile);
    png_set_IHDR(png, info, WIDTH, HEIGHT, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png, info);

    for (int i = 0; i < HEIGHT; i++) {
        row = (png_bytep)&pixels[i];
        png_write_row(png, row);
    }

    png_write_end(png, NULL);
    fclose(outputFile);
    png_destroy_write_struct(&png, &info);

    return 0;
}