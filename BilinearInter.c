#include <stdio.h>
#include <math.h>
#define SRCSIZE 3
#define TARSIZE 6
#define RA
// Out of Bound return 0, else return real value
static float GetPixel(int x, int y, int width, int height, float *Src){
    if(x < 0 || x >= width || y < 0 || y >= height) return 0.f;
    return *(Src + y*width + x);
}

static void BilinearInterpolate(float *Src, float *Tar){

    float tar_width = TARSIZE, tar_height = TARSIZE;
    float src_width = SRCSIZE, src_height = SRCSIZE;
    float r_ratio = src_height / tar_height;
    float c_ratio = src_width / tar_width;

    for(int r = 0 ; r < tar_height ; ++r){
        for(int c = 0 ; c < tar_width ; ++c, ++Tar){

            printf("========================\n");
            printf("Index %d, %d\n", r, c);
            printf("========================\n");

            float dc = (c + 0.5) * c_ratio - 0.5;
            float dr = (r + 0.5) * r_ratio - 0.5;

            printf("Before %2f, %2f\n", dr, dc);

            int ic = floorf(dc), ir = floorf(dr);

            printf("%d, %d\n", ir, ic);

            dr = (dr < 0.f) ? 1.0f : dr - ir;
            dc = (dc < 0.f) ? 1.0f : dc - ic;
            
            printf("After %f, %f\n", dr, dc);

            *Tar =         dc *  dr * GetPixel(ic + 1, ir + 1, src_height, src_width, Src) + 
                     (1 - dc) *  dr * GetPixel(    ic, ir + 1, src_height, src_width, Src) +
                      dc * (1 - dr) * GetPixel(ic + 1,     ir, src_height, src_width, Src) +
                (1 - dc) * (1 - dr) * GetPixel(    ic,     ir, src_height, src_width, Src);
            /*
            printf("Top Left: %f\n",   GetPixel(    ic,     ir, src_height, src_width, Src));  
            printf("Top Right: %f\n",   GetPixel(ic + 1,    ir, src_height, src_width, Src)); 
            printf("%d %d \n", ic, ir+1);
            printf("Bottom Left: %f\n",GetPixel(    ic, ir + 1, src_height, src_width, Src));
            printf("%d %d \n", ic+1, ir+1);
            printf("Bottom Right: %f\n",GetPixel(ic + 1, ir + 1, src_height, src_width, Src));
            printf("\n");
            */
        }
    }
}

int main(){
    float src[SRCSIZE][SRCSIZE] = {1,2,3,4,5,6,7,8,9};
    float dst[TARSIZE][TARSIZE];
    float* ptr_src = &src[0][0];
    float* ptr_dst = &dst[0][0];
    printf("========================\n");
    BilinearInterpolate(ptr_src, ptr_dst);
    printf("========================\n");
    
    for(int i = 0 ; i < TARSIZE ; ++i){
        for(int j = 0 ; j < TARSIZE ; ++j){
            printf("%2f ", dst[i][j]);
        }
        printf("\n");
    }
    
}