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
void getLumaFrame(int** frame_mem, FILE* yuv_file, Parameters p){
    int count;
    for(int r=0; r<p.height;r++)
        for(int c=0; c<p.width;c++)
            count=fread(&(frame_mem[r][c]),1,1,yuv_file);

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
void reconstruct(int** rec_frame, int** ref_frame, int i, int j, Parameters p, BestResult* MV){
    for(int a=i; a<i+p.blockSize; a++)
        for(int b=j; b<j+p.blockSize; b++)
            if( (0<=a+MV->vec_x) && (a+MV->vec_x<p.height) && (0<=b+MV->vec_y) && (b+MV->vec_y<p.width) )
                rec_frame[a][b] = ref_frame[a+MV->vec_x][b+MV->vec_y];
}
/************************************************************************************/
unsigned long long computeResidue(int** res_frame, int** curr_frame, int** rec_frame, Parameters p){
    unsigned long long accumulatedDifference = 0;
    int difference;
    for(int a=0; a<p.height; a++)
        for(int b=0; b<p.width; b++){
            difference = curr_frame[a][b] - rec_frame[a][b];
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
void getBlock(int* block, int** frame, int i, int j, Parameters p){
    for(int m=0; m<p.blockSize; m++)
        for(int n=0; n<p.blockSize; n++)
            block[m * p.blockSize + n] = frame[i+m][j+n];
}    
/************************************************************************************/
void getSearchArea(int* searchArea, int** frame, int i, int j, Parameters p){
    for(int m=-p.searchRange; m<p.searchRange+p.blockSize; m++)
        for(int n=-p.searchRange; n<p.searchRange+p.blockSize; n++)
            if ( ((0<=(i+m)) && ((i+m)<p.height)) && ((0<=(j+n)) && ((j+n)<p.width)) )
                searchArea[(p.searchRange + m) * (2 * p.searchRange + p.blockSize) + (p.searchRange + n)] = frame[i+m][j+n];
            else
                searchArea[(p.searchRange + m) * (2 * p.searchRange + p.blockSize) + (p.searchRange + n)] = 0; 
}

bool verifyPsearch(int k, int m, int searchRange) {
    return (-searchRange <= k) && (k <= searchRange) &&
           (-searchRange <= m) && (m <= searchRange);
}

// Function to verify the other four conditions
bool verifyOtherConditions(int rowIdx, int colIdx, int posX, int posY, int height, int width) {
    return (0 <= (rowIdx + posX)) && ((rowIdx + posX) < height) &&
           (0 <= (colIdx + posY)) && ((colIdx + posY) < width);
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
                sad += abs(CurrentBlock[i*p.blockSize+j] - SearchArea[(posX+i)*(2*p.searchRange+p.blockSize)+j+posY]);
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
__global__ void SAD_kernel(BestResult* bestResult, int* currentBlock, int* searchArea,int k, int m, Parameters p) {

    __shared__ int returnBlock[BLOCK_SIZE];
    unsigned int column = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int i;

    int posX = p.searchRange+k; 
    int posY = p.searchRange+m;

    
        // computes SAD disparity, by comparing the current block with the reference block at (k,m)
    returnBlock[column] = 0;
    for(i=0; i<p.blockSize;i++){
        returnBlock[column] += abs(currentBlock[i*p.blockSize+column] - searchArea[(posX+i)*(2*p.searchRange+p.blockSize)+column+posY]);
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
            bestResult->vec_x = k;
            bestResult->vec_y = m;
        }
    }
    
}

/************************************************************************************/
void StepSearch(BestResult* bestResult, int* CurrentBlock, int* SearchArea, int rowIdx, int colIdx, Parameters p){
    
    bestResult->sad = BigSAD; 
    bestResult->vec_x = 0;
    bestResult->vec_y = 0;

    // First prediction, at the center of the search area
    int CenterX=0;
    int CenterY=0;
    SAD(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, CenterX, CenterY, p);

    // Furthest search center
    int Distance = (p.searchRange)>>1;  // Initial distance = search range/2
    while (Distance >= 1) {
        SAD(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, CenterX-Distance, CenterY-Distance, p); // Top-Left
        SAD(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, CenterX-Distance, CenterY+0,        p); // Top-Center
        SAD(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, CenterX-Distance, CenterY+Distance, p); // Top-Right
        SAD(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, CenterX+0       , CenterY-Distance, p); // Center-Left
        SAD(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, CenterX+0       , CenterY+Distance, p); // Center-Right
        SAD(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, CenterX+Distance, CenterY-Distance, p); // Top-Left
        SAD(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, CenterX+Distance, CenterY+0,        p); // Top-Center
        SAD(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, CenterX+Distance, CenterY+Distance, p); // Top-Right
        // At this point, (bestResult->vec_x,bestResult->vec_y) marks the best search point and will be considered as the next search center
        CenterX = bestResult->vec_x;
        CenterY = bestResult->vec_y;
        // Divides the search distance by 2
        Distance >>= 1;
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
void xTZ8PointDiamondSearch(BestResult* bestResult, int* CurrentBlock, int* SearchArea, int rowIdx, int colIdx, int centroX, int centroY, int iDist, Parameters p){
    BestResult localBest;
    localBest.sad = bestResult->sad;
    localBest.bestDist = iDist;
    localBest.vec_x = 0;
    localBest.vec_y = 0;

    int SizeCurrentBlock = p.blockSize * p.blockSize;

    int SizeSearchArea = (2 * p.searchRange + p.blockSize) * (2 * p.searchRange + p.blockSize);
   

    int *d_CurrentBlock, *d_SearchArea;
    BestResult* d_bestResult;
    int *d_returnBlock;
    cudaMallocCheck(&d_CurrentBlock, SizeCurrentBlock * sizeof(int), "Failed to allocate memory for d_CurrentBlock");
    cudaMallocCheck(&d_SearchArea, SizeSearchArea * sizeof(int), "Failed to allocate memory for d_SearchArea");
    cudaError_t cudaStatus = cudaMalloc(&d_bestResult, sizeof(BestResult));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed: %s\n", "Failed to allocate memory for d_bestresult");
        fprintf(stderr, "CUDA error message: %s\n", cudaGetErrorString(cudaStatus));
        exit(EXIT_FAILURE);
    }
   

    // Copy data from Host to Device
    cudaMemcpyCheck(d_CurrentBlock, CurrentBlock, SizeCurrentBlock * sizeof(int), cudaMemcpyHostToDevice, "Failed to copy data to the device for d_CurrentBlock");
    cudaMemcpyCheck(d_SearchArea, SearchArea, SizeSearchArea * sizeof(int), cudaMemcpyHostToDevice, "Failed to copy data to the device for d_SearchArea");
    cudaMemcpyCheck(d_bestResult, bestResult, sizeof(BestResult), cudaMemcpyHostToDevice, "Failed to copy data to the device for d_bestResult");
    
    // Determine kernel launch configuration
    int threads_per_block = 32;
    dim3 blockDist(threads_per_block, 1, 1);
    dim3 gridDist(1, 1, 1);
   
    if ( iDist == 1 ){
        
            //SAD_kernel<<<gridDist, blockDist>>>(d_bestResult, d_CurrentBlock, d_SearchArea, rowIdx, colIdx, centroX - iDist, centroY, p);
            //cudaMemcpyCheck(bestResult, d_bestResult , sizeof(BestResult), cudaMemcpyDeviceToHost, "FAILED TO COPY DATA TO THE DEVICE");

            cudaError_t cudaStatus2 = cudaMemcpy(&(localBest), d_bestResult, sizeof(BestResult), cudaMemcpyDeviceToHost);
            if (cudaStatus2 != cudaSuccess) {
                fprintf(stderr, "CUDA memcpy failed: %s\n", "FAILED TO COPY DATA TO THE DEVICE");
                fprintf(stderr, "CUDA error message: %s\n", cudaGetErrorString(cudaStatus2));
                exit(EXIT_FAILURE);
            }
        
            //SAD( &(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, centroX-iDist, centroY      , p);
        
        
        SAD( &(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, centroX      , centroY-iDist, p);
        SAD( &(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, centroX      , centroY+iDist, p);
        SAD( &(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, centroX+iDist, centroY      , p);
    }else{
        int iTop        = centroY - iDist;
        int iBottom     = centroY + iDist;
        int iLeft       = centroX - iDist;
        int iRight      = centroX + iDist;
        if ( iDist <= 8 ){
            int iTop_2     = centroY - (iDist>>1);
            int iBottom_2  = centroY + (iDist>>1);
            int iLeft_2    = centroX - (iDist>>1);
            int iRight_2   = centroX + (iDist>>1);
            SAD( &(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, centroX,  iTop,    p);
            SAD( &(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, iLeft,    centroY, p);
            SAD( &(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, iRight,   centroY, p);
            SAD( &(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, centroX,  iBottom, p);
            SAD( &(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, iLeft_2,  iTop_2,  p);
            SAD( &(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, iRight_2, iTop_2,  p);
            SAD( &(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, iLeft_2,  iBottom_2, p);
            SAD( &(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, iRight_2, iBottom_2, p);
        }
        else{
            SAD( &(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, centroX, iTop,    p);
            SAD( &(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, iLeft,   centroY, p);
            SAD( &(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, iRight,  centroY, p);
            SAD( &(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, centroX, iBottom, p);
            for(int index=1; index<4; index++){
                int iPosYT     = iTop    + ((iDist>>2) * index);
                int iPosYB     = iBottom - ((iDist>>2) * index);
                int iPosXL     = centroX - ((iDist>>2) * index);
                int iPosXR     = centroX + ((iDist>>2) * index);
                SAD( &(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, iPosXL, iPosYT, p);
                SAD( &(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, iPosXR, iPosYT, p);
                SAD( &(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, iPosXL, iPosYB, p);
                SAD( &(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, iPosXR, iPosYB, p);
            }
        }
    }
    if (localBest.sad < bestResult->sad){
        bestResult->sad = localBest.sad;
        bestResult->bestDist = localBest.bestDist;
        bestResult->vec_x = localBest.vec_x;
        bestResult->vec_y = localBest.vec_y;
    }

    cudaFree(d_CurrentBlock);
    cudaFree(d_SearchArea);
    cudaFree(d_bestResult);

}
/************************************************************************************/
void TZSearch(BestResult* bestResult, int* CurrentBlock, int* SearchArea, int rowIdx, int colIdx, Parameters p){
    int bestX, bestY;
    bestResult->sad = BigSAD; 
    bestResult->bestDist = 0;
    bestResult->vec_x = 0;
    bestResult->vec_y = 0;

    // First prediction, at the center of the search area
    SAD(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, 0, 0, p);

    // Initial Search: iDist in [1, 2, 4, 8, 16, 32, 64]
    int iDist = 1;
    while (iDist <= p.searchRange) {
        xTZ8PointDiamondSearch(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, 0, 0, iDist, p);
        iDist <<= 1;
    }

    // Raster Search
    bestX = bestResult->vec_x;
    bestY = bestResult->vec_y;
    if ((bestX > p.iRaster) || (bestY > p.iRaster) || (-bestX > p.iRaster) || (-bestY > p.iRaster)){
        int Top = -(int)(p.searchRange/2);
        int Bottom = (int)(p.searchRange/2);
        int Left = -(int)(p.searchRange/2);
        int Right = (int)(p.searchRange/2);
        for(int iStartY=Top; iStartY<Bottom; iStartY+=p.iRaster)
            for(int iStartX=Left; iStartX<Right; iStartX+=p.iRaster)
                SAD(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, iStartX, iStartY, p);
    }

    // Refinement
    bestX = bestResult->vec_x;
    bestY = bestResult->vec_y;
    int RefinementCount=0;
    if ((bestX != 0) || (bestY != 0))
        while ((bestResult->vec_x == bestX) && (bestResult->vec_y == bestY)){
            iDist = 1;
            while (iDist <= p.searchRange) {
                xTZ8PointDiamondSearch(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, bestX, bestY, iDist, p);
            
                if (((4 <= iDist) && (bestResult->bestDist == 0)) ||
                    ((8 <= iDist) && (bestResult->bestDist <= 1)) ||
                    ((16 <= iDist) && (bestResult->bestDist <= 2)) ||
                    ((32 <= iDist) && (bestResult->bestDist <= 4)))
                        break;

                iDist <<= 1;
            }
            if (((bestResult->vec_x == bestX) && (bestResult->vec_y == bestY)) || (RefinementCount == 7))
                break;
            else{
                bestX = bestResult->vec_x;
                bestY = bestResult->vec_y;
                RefinementCount += 1;
            }
        }
}
/************************************************************************************/
void fullSearch(BestResult* bestResult, int* CurrentBlock, int* SearchArea, int rowIdx, int colIdx, Parameters p){
    bestResult->sad = BigSAD;
    bestResult->bestDist = 0;
    bestResult->vec_x = 0;
    bestResult->vec_y = 0;

    int SizeCurrentBlock = p.blockSize * p.blockSize;
    int SizeSearchArea = (2 * p.searchRange + p.blockSize) * (2 * p.searchRange + p.blockSize);

    int *d_CurrentBlock, *d_SearchArea;
    BestResult* d_bestResult;

    cudaMallocCheck(&d_CurrentBlock, SizeCurrentBlock * sizeof(int), "Failed to allocate memory for d_CurrentBlock");
    cudaMallocCheck(&d_SearchArea, SizeSearchArea * sizeof(int), "Failed to allocate memory for d_SearchArea");
    cudaError_t cudaStatus = cudaMalloc(&d_bestResult, sizeof(BestResult));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed: %s\n", "Failed to allocate memory for d_bestresult");
        fprintf(stderr, "CUDA error message: %s\n", cudaGetErrorString(cudaStatus));
        exit(EXIT_FAILURE);
    }
    

    // Copy data from Host to Device
    cudaMemcpyCheck(d_CurrentBlock, CurrentBlock, SizeCurrentBlock * sizeof(int), cudaMemcpyHostToDevice, "Failed to copy data to the device for d_CurrentBlock");
    //cudaMemcpyCheck(d_SearchArea, SearchArea, SizeSearchArea * sizeof(int), cudaMemcpyHostToDevice, "Failed to copy data to the device for d_SearchArea");
    cudaMemcpyCheck(d_bestResult, bestResult, sizeof(BestResult), cudaMemcpyHostToDevice, "Failed to copy data to the device for d_bestResult");
    
    // Determine kernel launch configuration
    int threads_per_block = 32;
    dim3 blockDist(threads_per_block, 1, 1);
    dim3 gridDist(1, 1, 1);

    int posX ; 
    int posY ;
    int offset=0;
    
    
    for(int iStartX=-p.searchRange; iStartX<p.searchRange; iStartX++){

        posX = p.searchRange+iStartX;
        if (0 <= (rowIdx+posX) && (rowIdx+posX) < p.height){

            for(int iStartY=-p.searchRange; iStartY<p.searchRange; iStartY++){

                posY = p.searchRange+iStartY;
                if (0 <= (colIdx+posY) && (colIdx+posY) < p.width){
                    // Launch the kernel with the required portion of the searchArea matrix
                    SAD_kernel<<<gridDist, blockDist>>>(d_bestResult, d_CurrentBlock, d_SearchArea,iStartX, iStartY, p);


                    //SAD_kernel<<<gridDist, blockDist>>>(d_bestResult, d_CurrentBlock, d_SearchArea,iStartX, iStartY, p);
                }    
            }

        }
    }
    cudaError_t cudaStatus2 = cudaMemcpy(bestResult, d_bestResult, sizeof(BestResult), cudaMemcpyDeviceToHost);
    if (cudaStatus2 != cudaSuccess) {
        fprintf(stderr, "CUDA memcpy failed: %s\n", "FAILED TO COPY DATA TO THE DEVICE");
        fprintf(stderr, "CUDA error message: %s\n", cudaGetErrorString(cudaStatus2));
        exit(EXIT_FAILURE);
    }
    cudaFree(d_CurrentBlock);
    cudaFree(d_SearchArea);
    cudaFree(d_bestResult);
}
/************************************************************************************/
void MotionEstimation(BestResult** motionVectors, int **curr_frame, int **ref_frame, Parameters p){
    BestResult* bestResult;

    int* CurrentBlock  = (int*)malloc(p.blockSize*p.blockSize * sizeof(int));
    int* SearchArea = (int*)malloc((2*p.searchRange+p.blockSize) * (2*p.searchRange+p.blockSize) * sizeof(int));
 

	for(int rowIdx=0; rowIdx<(p.height-p.blockSize+1); rowIdx+=p.blockSize)
		for(int colIdx=0; colIdx<(p.width-p.blockSize+1); colIdx+=p.blockSize){
			// Gets current block and search area data
	        getBlock(CurrentBlock, curr_frame, rowIdx, colIdx, p);
	        getSearchArea(SearchArea, ref_frame, rowIdx, colIdx, p);
            bestResult = &(motionVectors[rowIdx / p.blockSize][colIdx / p.blockSize]);
            // Runs the motion estimation algorithm on this block
            switch (p.algorithm)
            {
            case FSBM:
                fullSearch(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, p);
                break;
            case TZS:
                TZSearch(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, p);
                break;
            case SS:
                StepSearch(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, p);
                break;
            default:
                break;
            }
        }

    free(CurrentBlock);
    free(SearchArea);
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
    int** curr_frame = (int**)malloc(p.height * sizeof(int*));
    int** ref_frame  = (int**)malloc(p.height * sizeof(int*));
    int** res_frame  = (int**)malloc(p.height * sizeof(int*));
    int** rec_frame  = (int**)malloc(p.height * sizeof(int*));
    for (int i = 0; i < p.height; i++){
        curr_frame[i] = (int*)malloc(p.width * sizeof(int));
        ref_frame[i]  = (int*)malloc(p.width * sizeof(int));
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
    //
    for (int frameNum = 0; frameNum < p.frames; frameNum++) {
        int** temp;
        temp = ref_frame;
        ref_frame = curr_frame;                // ref_frame contains the previous (reference) luminance frame
        curr_frame = temp;
        getLumaFrame(curr_frame, video_in, p); // curr_frame contains the current luminance frame

        // Process the current frame, one block at a time, to obatin an array with the motion vectors and SAD values
        MotionEstimation(motionVectors, curr_frame, ref_frame, p);

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
                    rec_frame[r][c] = ref_frame[r][c];


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
        free(curr_frame[i]);
        free(ref_frame[i]);
        free(res_frame[i]);
        free(rec_frame[i]);
    }
    free(curr_frame);
    free(ref_frame);
    free(res_frame);
    free(rec_frame);

    return 0;
}




