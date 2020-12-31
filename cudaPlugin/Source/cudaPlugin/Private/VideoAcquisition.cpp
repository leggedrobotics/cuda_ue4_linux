// Fill out your copyright notice in the Description page of Project Settings.


#include "VideoAcquisition.h"

#include "VulkanResources.h"

// Sets default values for this component's properties
AVideoAcquisition::AVideoAcquisition()
{
	// Set this component to be initialized when the game starts, and to be ticked every frame.  You can turn these features
	// off to improve performance if you don't need them.
	//PrimaryComponentTick.bCanEverTick = false;
	PrimaryActorTick.bCanEverTick = true;
}


AVideoAcquisition::~AVideoAcquisition()
{

}

// Called when the game starts
void AVideoAcquisition::BeginPlay()
{
	Super::BeginPlay();

	RenderTarget->InitCustomFormat(VideoWidth, VideoHeight, PF_B8G8R8A8, true);
	RenderTarget->bAutoGenerateMips = 0;
	RenderTarget->bForceLinearGamma = 0;

    InitCamera();


    // Start the Ximea Camera acquisition loop
    if (cameraThread == nullptr) {

                  cameraThread = new std::thread([&](){
                          threadLoop();
                  });
     }


}

void AVideoAcquisition::EndPlay(const EEndPlayReason::Type)
{


    stopped = true;



}


void AVideoAcquisition::Tick(float DeltaTime)
{

}



void AVideoAcquisition::InitCamera()
{


    // Read pre-recorded raw camera images
    std::ifstream infile ("/home/heapvrar/rawbayer.data",std::ifstream::binary);

    // allocate memory for temporary file content (Bayer pattern images)
    char* buffer = new char[VideoWidth*VideoHeight];

    for(int i=0;i<240;i++){


          std::cout <<"DEBUG -- Loading Frame ID: " << i << std::endl;
          // read content of infile
          infile.read(buffer,VideoWidth*VideoHeight);

          // Allocate GPU memory
          dpFrameBayer[i] = nppiMalloc_8u_C1(VideoWidth, VideoHeight, &(nPitchBayer));

          // Copy image from Host to GPU memory
          cudaMemcpy2D(dpFrameBayer[i],VideoWidth,buffer,nPitchBayer,(size_t)VideoWidth,(size_t)VideoHeight,cudaMemcpyHostToDevice);


    }

    infile.close();


}


void AVideoAcquisition::threadLoop()
{

    printf("Starting camera acquisition...\n");


    int CurrTS_sec;
    int CurrTS_usec;
    unsigned int frameIndex = 0;

    NppiSize osize;
    osize.width = VideoWidth;
    osize.height = VideoHeight;

    NppiRect orect;
    orect.x = 0;
    orect.y = 0;
    orect.width = VideoWidth;
    orect.height = VideoHeight;

    NppStatus status;

    // allocated memory for the RGBA image
    dpFrame = nppiMalloc_8u_C4(VideoWidth, VideoHeight, &(nPitch));


    int CurrentFrameNumber = 0;
    int CurrentFrameID = 0;


    while(!stopped) {


        CurrentFrameID = CurrentFrameNumber++%240;

        // Perform Bayer to RGB conversion
        status = nppiCFAToRGBA_8u_C1AC4R(dpFrameBayer[CurrentFrameID], VideoWidth, osize, orect, dpFrame, nPitch,NPPI_BAYER_BGGR, NPPI_INTER_UNDEFINED,255); //NPPI_INTER_CUBIC, NPPI_INTER_UNDEFINED

        std::cout <<"DEBUG -- Frame ID: " << CurrentFrameID << " RGB status " << status <<  " vidBayerPitch: " << nPitchBayer <<  " vidPitch: " << nPitch <<std::endl;


    }// end of thread while loop


    printf("The camera acquisiton is closed successfully!\n");
    nppiFree(dpFrame);



}// end of threadLoop function


void AVideoAcquisition::UpdateTextureFromGPU()
{
    //CUDA Expects the resources to be in an array
    Resources[0] = CudaResource;
    AVideoAcquisition* This = this;

    //Texture update must be executed on the Render Thread
    ENQUEUE_RENDER_COMMAND(UpdateTexture)(
        [This](FRHICommandListImmediate& RHICmdList)
    {
        cudaError_t err0 = cudaGraphicsMapResources(1, This->Resources, 0);
        cudaArray *CuArray;
        cudaError_t err1 = cudaGraphicsSubResourceGetMappedArray(&CuArray, This->CudaResource, 0, 0);
        cudaError_t err2 = cudaMemcpy2DToArray(
            CuArray, // dst array
            0, 0,    // offset
            (void*)(This->dpFrame), This->nPitch,       // src
            This->VideoWidth * 4, This->VideoHeight, // extent
            cudaMemcpyDeviceToDevice); // kind

        cudaError_t err3 = cudaGraphicsUnmapResources(1, This->Resources, 0);

        if (!(err0 == cudaSuccess && err1 == cudaSuccess && err2 == cudaSuccess && err3 == cudaSuccess)) {
            //UE_LOG(LogGPUVideoDecode, Warning, TEXT("Failed to update texture: %d %d %d %d"), (int)err0, (int)err1, (int)err2, (int)err3);
        }
        else {
            //UE_LOG(LogGPUVideoDecode, Error, TEXT("Texture Update Okay"));
        }
    });
}



