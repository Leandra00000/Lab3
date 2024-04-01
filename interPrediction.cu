/************************************************************************/
/* File: interPrediction.c                                              */
/* Author: Nuno Roma <Nuno.Roma@tecnico.ulisboa.pt                      */
/* Date: February 23th, 2024                                            */
/************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define FSBM 0                // Full-Search Block Matching (FSBM) motion estimation algorithm
#define SS   1                // (Three/Four) Step-Search (SS) block Matching motion estimation algorithm
#define TZS  2                // Test Zonal Search (TZS) block matching motion estimation algorithm

#define SEARCH_RANGE 64      // Search range (at each direction)
#define BLOCK_SIZE   32      // Block size (at each direction)
#define iRASTER       5      // TZS iRaster parameter

#define BigSAD 999999         // it could be any other big integer...

int flag=0;
typedef struct {
    char *video_name; // YUV input file
    int width;        // luminance width
    int height;       // luminance height
    int frames;       // number of frames to process
    int algorithm;    // motion estimation algorithm
    int searchRange;  // search range (at each direction)
    int blockSize;    // block size (at each direction)
    int iRaster;      // TZS iRaster parameter
    int debug;        // verbose mode
} Parameters;

typedef struct {
    int vec_x;
    int vec_y;
    int sad;
    int bestDist;
} BestResult;


/************************************************************************************/
void getLumaFrame(int* frame_mem, FILE* yuv_file, Parameters p){
    int count;
    for(int r=0; r<p.height;r++)
        for(int c=0; c<p.width;c++)
           count=fread(&(frame_mem[r*p.width+c]),1,1,yuv_file);

    // Skips the color Cb and Cr components in the YUV 4:2:0 file
    fseek(yuv_file,p.width*p.height/2,SEEK_CUR);
}
/************************************************************************************/
void setLumaFrame(int** frame_mem, FILE* yuv_file, Parameters p){
    __uint8_t temp;
    for(int r=0; r<p.height; r++)
        for(int c=0; c<p.width; c++){
            temp=(__uint8_t)frame_mem[r][c];
            fwrite(&temp,1,1,yuv_file);
        }
    // writes 2*(height/2)*(width/2) values to fill in chrominance part with 128
    temp=(__uint8_t)128;
    for(int r=0; r<p.height/2; r++)
        for(int c=0; c<p.width; c++){
            fwrite(&temp,1,1,yuv_file);
        }
}
/************************************************************************************/
void reconstruct(int** rec_frame, int* ref_frame, int i, int j, Parameters p, BestResult* MV){
    for(int a=i; a<i+p.blockSize; a++)
        for(int b=j; b<j+p.blockSize; b++)
            if( (0<=a+MV->vec_x) && (a+MV->vec_x<p.height) && (0<=b+MV->vec_y) && (b+MV->vec_y<p.width) )
                rec_frame[a][b] = ref_frame[(a+MV->vec_x)*p.width+b+MV->vec_y];
}
/************************************************************************************/
unsigned long long computeResidue(int** res_frame, int* curr_frame, int** rec_frame, Parameters p){
    unsigned long long accumulatedDifference = 0;
    int difference;
    for(int a=0; a<p.height; a++)
        for(int b=0; b<p.width; b++){
            difference = curr_frame[a*p.width+b] - rec_frame[a][b];
            if (difference < 0) 
                difference = - difference;
            if (255 < difference)
                difference = 255;
            res_frame[a][b] = difference;
            accumulatedDifference += difference;
        }
    return(accumulatedDifference);
}
/************************************************************************************/
void getBlock(int* block, int* frame, int i, int j, Parameters p){
    for(int m=0; m<p.blockSize; m++)
        for(int n=0; n<p.blockSize; n++)
            block[m*p.blockSize+n] = frame[(i+m)*p.width+j+n];
}    
/************************************************************************************/
void getSearchArea(int* searchArea, int* frame, int i, int j, Parameters p){
    for(int m=-p.searchRange; m<p.searchRange+p.blockSize; m++)
        for(int n=-p.searchRange; n<p.searchRange+p.blockSize; n++)
            if ( ((0<=(i+m)) && ((i+m)<p.height)) && ((0<=(j+n)) && ((j+n)<p.width)) )
                searchArea[(p.searchRange + m) * (2 * p.searchRange + p.blockSize) + (p.searchRange + n)] = frame[(i+m)*p.width+j+n];
            else
                searchArea[(p.searchRange + m) * (2 * p.searchRange + p.blockSize) + (p.searchRange + n)] = 0; 
}
/************************************************************************************/
void SAD(BestResult* bestResult, int* CurrentBlock, int* SearchArea, int rowIdx, int colIdx, int k, int m, Parameters p){
    // k, m: displacement (motion vector) under analysis (in the search area)

    int sad = 0;
    int posX = p.searchRange+k; // normalized coordinates within search area, between 0 and 2*searchRange
    int posY = p.searchRange+m; // normalized coordinates within search area, between 0 and 2*searchRange
    // checks if search area range is valid (inside frame borders) and if current block range is valid (inside frame borders)
    if ( (-p.searchRange <= k) && (k <= p.searchRange) && \
         (-p.searchRange <= m) && (m <= p.searchRange) && \
         (0 <= (rowIdx+posX)) && ((rowIdx+posX) < p.height) && \
         (0 <= (colIdx+posY)) && ((colIdx+posY) < p.width) ){
        // computes SAD disparity, by comparing the current block with the reference block at (k,m)
        
        for(int i=0; i<p.blockSize; i++){
            for(int j=0; j<p.blockSize; j++){
                if ( ((0<=(rowIdx+k+i)) && ((0<=(colIdx+m+j) ))))
                    sad += abs(CurrentBlock[(i+rowIdx)*p.width+(j+colIdx)] - SearchArea[(k+i+rowIdx)*p.width+j+m+colIdx]);
                else
                    sad+= abs(CurrentBlock[(i+rowIdx)*p.width+(j+colIdx)]);
                
                
            }
        }
        // compares the obtained sad with the best so far for that block
        if (sad < bestResult->sad){
            bestResult->sad = sad;
            bestResult->vec_x = k;
            bestResult->vec_y = m;
        }
    }
}

/************************************************************************************/
__global__ void SAD_kernel(BestResult* bestResult, int* currentBlock, int* searchArea,int rowIdx, int colIdx, Parameters p) {


    __shared__ int returnBlock[BLOCK_SIZE];
    unsigned int i;
    unsigned int column = blockIdx.x*blockDim.x + threadIdx.x;
    for(int iStartX=-p.searchRange; iStartX<p.searchRange; iStartX++){
        for (int iStartY = -p.searchRange; iStartY < p.searchRange; iStartY ++){
            
            returnBlock[column] = 0;
            for(i=0; i<p.blockSize;i++){
                returnBlock[column] += abs(currentBlock[(i+rowIdx)*p.width+(column+colIdx)] - searchArea[(iStartX+i+rowIdx)*p.width+column+iStartY+colIdx]);
            }

            unsigned int tid = column;
            for (i=blockDim.x >> 1; i > 0; i = i >> 1) {
                if(tid < i){
                    returnBlock[tid] += returnBlock[tid + i];
                }
                __syncthreads();
            }
                

            if(tid == 0){
                if (returnBlock[0] < bestResult->sad){
                    bestResult->sad = returnBlock[0];
                    bestResult->vec_x = iStartX;
                    bestResult->vec_y = iStartY;
                }
            }
            

        }
        
    }
    
     
    
    
}


/**********************************************************************************/
BestResult* allocateBestResultOnDevice() {
    BestResult* device_bestResult;
    cudaMalloc((void**)&device_bestResult, sizeof(BestResult));
    return device_bestResult;
}
/**/

void cudaMallocCheck(int** devPtr, size_t size, const char* errorMsg) {
    cudaError_t cudaStatus = cudaMalloc(devPtr, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed: %s\n", errorMsg);
        fprintf(stderr, "CUDA error message: %s\n", cudaGetErrorString(cudaStatus));
        exit(EXIT_FAILURE);
    }
}

void cudaMemcpyCheck(void* dst, const void* src, size_t count, cudaMemcpyKind kind, const char* errorMsg) {
    cudaError_t cudaStatus = cudaMemcpy(dst, src, count, kind);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA memcpy failed: %s\n", errorMsg);
        fprintf(stderr, "CUDA error message: %s\n", cudaGetErrorString(cudaStatus));
        exit(EXIT_FAILURE);
    }
}

/************************************************************************************/
void fullSearch(BestResult* bestResult, int* CurrentBlock, int* SearchArea,  int *curr_frame, int *ref_frame,int rowIdx, int colIdx, Parameters p){
    bestResult->sad = BigSAD;
    bestResult->bestDist = 0;
    bestResult->vec_x = 0;
    bestResult->vec_y = 0;
    
    BestResult* d_bestResult;

    
    if (! (0<=rowIdx-p.searchRange && 0<=colIdx-p.searchRange && (rowIdx+p.searchRange+p.searchRange) < p.height && (colIdx+p.searchRange+p.searchRange) < p.width)){
        for(int iStartX=-p.searchRange; iStartX<p.searchRange; iStartX++){

            int posX = p.searchRange+iStartX;
            if ((rowIdx+posX) < p.height){

                for(int iStartY=-p.searchRange; iStartY<p.searchRange; iStartY++){

                    int posY = p.searchRange+iStartY;
                    if ((colIdx+posY) < p.width){
                        SAD(bestResult, curr_frame, ref_frame, rowIdx, colIdx, iStartX, iStartY, p);
                    }    
                }

            }
        }
    }else{
        cudaError_t cudaStatus = cudaMalloc(&d_bestResult, sizeof(BestResult));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "CUDA malloc failed: %s\n", "Failed to allocate memory for d_bestresult");
            fprintf(stderr, "CUDA error message: %s\n", cudaGetErrorString(cudaStatus));
            exit(EXIT_FAILURE);
        }
        

        cudaMemcpyCheck(d_bestResult, bestResult, sizeof(BestResult), cudaMemcpyHostToDevice, "Failed to copy data to the device for d_bestResult");
    
        int threads_per_block = 32;
        dim3 blockDist(threads_per_block, 1, 1);
        dim3 gridDist(1, 1, 1);
        /*
        for(int iStartX=-p.searchRange; iStartX<p.searchRange; iStartX++){

            int posX = p.searchRange+iStartX;
            if (0 <= (rowIdx+posX) && (rowIdx+posX) < p.height){

                for(int iStartY=-p.searchRange; iStartY<p.searchRange; iStartY++){

                    int posY = p.searchRange+iStartY;
                    if (0 <= (colIdx+posY) && (colIdx+posY) < p.width){
                        // Launch the kernel with the required portion of the searchArea matrix
                        */
                        SAD_kernel<<<gridDist, blockDist>>>(d_bestResult, CurrentBlock, SearchArea,rowIdx, colIdx, p);
                        
                        cudaStatus = cudaGetLastError();
                        if (cudaStatus != cudaSuccess) {
                            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
                            fprintf(stderr,"%d rrow, %d col: ",rowIdx,colIdx);
                            exit(EXIT_FAILURE);
                        }

                        // Synchronize the device to catch any errors from the kernel
                        cudaStatus = cudaDeviceSynchronize();
                        if (cudaStatus != cudaSuccess) {
                            fprintf(stderr, "CUDA error during synchronization: %s\n", cudaGetErrorString(cudaStatus));
                            printf("%d rrow, %d col: ",rowIdx,colIdx);
                            exit(EXIT_FAILURE);
                        }
                        //SAD_kernel<<<gridDist, blockDist>>>(d_bestResult, d_CurrentBlock, d_SearchArea,iStartX, iStartY, p);
                        
                        //SAD(bestResult, curr_frame, ref_frame, rowIdx, colIdx, iStartX, iStartY, p);
                  /*
                    }    
                }

            }
        }
        */
        cudaError_t cudaStatus2 = cudaMemcpy(bestResult, d_bestResult, sizeof(BestResult), cudaMemcpyDeviceToHost);
        if (cudaStatus2 != cudaSuccess) {
            fprintf(stderr, "CUDA memcpy failed: %s\n", "FAILED TO COPY DATA TO THE DEVICE");
            fprintf(stderr, "CUDA error message: %s\n", cudaGetErrorString(cudaStatus2));
            exit(EXIT_FAILURE);
        }
        cudaFree(d_bestResult);
    }
    
    
}
/************************************************************************************/
void MotionEstimation(BestResult** motionVectors, int *d_curr_frame, int *d_ref_frame, int *curr_frame, int *ref_frame,  Parameters p){
    BestResult* bestResult;

    int* CurrentBlock; // = (int*)malloc(p.blockSize*p.blockSize * sizeof(int));
    int* SearchArea; //= (int*)malloc((2*p.searchRange+p.blockSize) * (2*p.searchRange+p.blockSize) * sizeof(int));
 

	for(int rowIdx=0; rowIdx<(p.height-p.blockSize+1); rowIdx+=p.blockSize)
		for(int colIdx=0; colIdx<(p.width-p.blockSize+1); colIdx+=p.blockSize){
			// Gets current block and search area dat
            // Runs the motion estimation algorithm on this block
            bestResult = &(motionVectors[rowIdx / p.blockSize][colIdx / p.blockSize]);
            switch (p.algorithm)
            {
            case FSBM:
                fullSearch(bestResult, d_curr_frame, d_ref_frame, curr_frame, ref_frame, rowIdx, colIdx, p);
                break;
            default:
                break;
            }
        }

}
/************************************************************************************/
/************************************************************************************/
int main(int argc, char** argv) {
    
    struct timespec t0,t1;
    unsigned long long accumulatedResidue = 0;    
    
    // Read input parameters
    if (argc != 7) {
        printf("USAGE: %s <videoPath> <Width> <Height> <NFrames> <ME Algorithm: 0=FSBM; 1=SS; 2=TZS> <Debug Mode: 0=silent; 1=verbose>\n",argv[0]);
        exit(1);
    }
    Parameters p;
    p.video_name = argv[1];
    p.width = atoi(argv[2]);
    p.height = atoi(argv[3]);
    p.frames = atoi(argv[4]);
    p.algorithm = atoi(argv[5]);
    p.searchRange = SEARCH_RANGE; // Search range (at each direction)
    p.blockSize = BLOCK_SIZE;     // Block size (at each direction)
    p.iRaster = iRASTER;          // TZS iRaster parameter
    p.debug = atoi(argv[6]);

    switch (p.algorithm)
    {
    case FSBM:
        printf("Running FSBM algorithm\n"); 
        break;
    case SS:
        printf("Running Step-Search algorithm\n"); 
        break;
    case TZS:
        printf("Running TZS algorithm\n"); 
        break;
    default:
        printf("ERROR: Invalid motion estimation algorithm\n");
        exit(-1);
    }

    // Video files
    FILE *video_in;
    FILE *residue_out;
    FILE *reconst_out;
    video_in = fopen(p.video_name, "rb");
    residue_out = fopen("residue.yuv", "wb");
    reconst_out = fopen("reconst.yuv", "wb");
    if (!video_in || !residue_out || !reconst_out) {
        printf("Opening input/output file error\n");
        exit(1);
    }

    // Frame memory allocation
    int* curr_frame = (int*)malloc(p.height*p.width * sizeof(int));
    int* ref_frame  = (int*)malloc(p.height*p.width * sizeof(int));
    int** res_frame  = (int**)malloc(p.height * sizeof(int*));
    int** rec_frame  = (int**)malloc(p.height * sizeof(int*));
    for (int i = 0; i < p.height; i++){
        res_frame[i]  = (int*)malloc(p.width * sizeof(int));
        rec_frame[i]  = (int*)malloc(p.width * sizeof(int));
    }

    // Memory allocation of result table
    BestResult** motionVectors = (BestResult**)malloc(p.height/p.blockSize * sizeof(BestResult*));
    for (int i = 0; i < p.height/p.blockSize; i++)
        motionVectors[i] = (BestResult*)malloc(p.width/p.blockSize * sizeof(BestResult));
    BestResult* MV;

    clock_gettime(CLOCK_REALTIME, &t0);
    // Read first frame
    getLumaFrame(curr_frame, video_in, p);      // curr_frame contains the current luminance frame

    int * d_curr_frame, *d_ref_frame;
    cudaMallocCheck(&d_curr_frame, p.width*p.height * sizeof(int), "Failed to allocate memory for d_CurrentBlock");
    cudaMallocCheck(&d_ref_frame, p.width*p.height * sizeof(int), "Failed to allocate memory for d_SearchArea");

   
    //
    for (int frameNum = 0; frameNum < p.frames; frameNum++) {
        int* temp;
        
        temp = ref_frame;
        ref_frame = curr_frame;                // ref_frame contains the previous (reference) luminance frame
        curr_frame = temp;

        getLumaFrame(curr_frame, video_in, p); // curr_frame contains the current luminance frame

        // Process the current frame, one block at a time, to obatin an array with the motion vectors and SAD values
        cudaMemcpyCheck(d_ref_frame, ref_frame, p.width*p.height * sizeof(int), cudaMemcpyHostToDevice, "Failed to copy data to the device for d_SearchArea");
        cudaMemcpyCheck(d_curr_frame, curr_frame, p.width*p.height * sizeof(int), cudaMemcpyHostToDevice, "Failed to copy data to the device for d_CurrentBlock");
        MotionEstimation(motionVectors, d_curr_frame, d_ref_frame, curr_frame, ref_frame, p);

        // Recustruct the predicted frame using the obtained motion vectors
        for (int rowIdx = 0; rowIdx < p.height - p.blockSize + 1; rowIdx += p.blockSize) {
    	    for (int colIdx = 0; colIdx < p.width - p.blockSize + 1; colIdx += p.blockSize) {
                // Gets best candidate block information
                MV = &( motionVectors[rowIdx / p.blockSize][colIdx / p.blockSize] );

                // Reconstructs current block using  the obtained motion estimation information
                reconstruct(rec_frame, ref_frame, rowIdx, colIdx, p, MV);

                // Print vector information
                if(p.debug)
                    printf("Frame %d : Block [%4d , %4d] = (%3d,%3d), SAD= %d\n", frameNum, colIdx, rowIdx, MV->vec_y, MV->vec_x, MV->sad);
            }
        }
        // Reconstructs borders of the frame not convered by motion estimation
        for(int r = 0; r < p.height; r++)
            for(int c = 0; c < p.width; c++)
                if(r > (p.height - p.blockSize + 1) || c > (p.width - p.blockSize + 1))
                    rec_frame[r][c] = ref_frame[r*p.width+c];


        // Compute residue block
        accumulatedResidue += computeResidue(res_frame, curr_frame, rec_frame, p);

        // Save reconstructed and residue frames
        setLumaFrame(rec_frame, reconst_out, p);
        setLumaFrame(res_frame, residue_out, p);
    }
    
    clock_gettime(CLOCK_REALTIME, &t1);
    
	printf ("%lf seconds elapsed \n", (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)*1e-9);
    printf ("Accumulated Residue = %llu \n", accumulatedResidue);

    // Frame memory free
    for (int i = 0; i < p.height; i++){
        free(res_frame[i]);
        free(rec_frame[i]);
    }
    free(curr_frame);
    free(ref_frame);
    free(res_frame);
    free(rec_frame);

    return 0;
}




