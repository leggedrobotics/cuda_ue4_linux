// Fill out your copyright notice in the Description page of Project Settings.


#include "VideoAcquisition.h"


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

    InitCUDA();

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
    /*
    GameEnded = true;

    //Cleanup allocated resources
    nppiFree(dpFrameMain);
    nppiFree(dpFrame);
    */
    /*
    cudaError_t x = cudaGraphicsUnregisterResource(CudaResource);
    if (x != cudaSuccess) {
        //UE_LOG(LogGPUVideoDecode, Error, TEXT("Failed to unregister Cuda Resource %s %d"), *VideoSourceTopic, x);
    }
    */
}


void AVideoAcquisition::Tick(float DeltaTime)
{

}


bool AVideoAcquisition::InitCUDA()
{
        //Initialize CUDA
        CUresult initRet = cuInit(0);
        int iGpu = 0;
        int nGpu = 0;
        CUresult getCountRet = cuDeviceGetCount(&nGpu);
        if (iGpu < 0 || iGpu >= nGpu)
        {
          //  UE_LOG(LogGPUVideoDecode, Error, TEXT("GPU ordinal out of range.Should be within[0, %d]"), nGpu - 1);
            return false;
        }
        CUdevice cuDevice = 0;
        CUresult devGetRet = cuDeviceGet(&cuDevice, iGpu);
        char szDeviceName[80];
        CUresult devGetNameRet = cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice);
       // UE_LOG(LogGPUVideoDecode, Log, TEXT("GPU in use: %s"), UTF8_TO_TCHAR(szDeviceName));
        CUresult ctxCreateRet = cuCtxCreate(&cuContext, 0, cuDevice);
        //UE_LOG(LogGPUVideoDecode, Log, TEXT("Decode with NvDecoder."));

        //Allocate GPU memory for intermediate steps
        //dpFrameMain = nppiMalloc_8u_C4(VideoWidth, VideoHeight, &nPitchMain);
        //dpFrame = nppiMalloc_8u_C4(VideoWidth, VideoHeight, &nPitch);

        //Register the texture on which the video is rendered as a CUDA resource
        AVideoAcquisition* This = this;

        //Texture registration must be executed on the Render Thread
        ENQUEUE_RENDER_COMMAND(RegisterResource)(
            [This](FRHICommandListImmediate& RHICmdList)
            {
                FD3D11TextureBase* D3D11Texture = GetD3D11TextureFromRHITexture(This->RenderTarget->Resource->TextureRHI);
                cudaError_t err = cudaGraphicsD3D11RegisterResource(&(This->CudaResource), D3D11Texture->GetResource(), cudaGraphicsRegisterFlagsNone);
                if (err != cudaSuccess) {
                  //  UE_LOG(LogGPUVideoDecode, Error, TEXT("Failed to register CUDA resource: %d"), (int)err);
                    return;
                }
            });
   
    return true;
}

void AVideoAcquisition::InitCamera()
{

    // Read pre-recorded raw camera images
    std::ifstream infile ("C:/Users/burak/Documents/rawbayer.data",std::ifstream::binary);

    // allocate memory for temporary file content (Bayer pattern images)
    char* buffer = new char[VideoWidth*VideoHeight];

    for(int i=0;i<240;i++){

          std::cout <<"DEBUG -- Loading Frame ID: " << i << std::endl;
          // read content of infile
          infile.read(buffer,VideoWidth*VideoHeight);

          // Allocate GPU memory
          dpFrameBayer[i] = nppiMalloc_8u_C1(VideoWidth, VideoHeight, &(nPitchBayer));

          // Copy image from Host to GPU memory
          cudaMemcpy2D(dpFrameBayer[i], nPitchBayer,buffer, VideoWidth,(size_t)VideoWidth,(size_t)VideoHeight,cudaMemcpyHostToDevice);

    }

    infile.close();


}


void AVideoAcquisition::threadLoop()
{

    printf("Starting camera acquisition...\n");

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
        status = nppiCFAToRGBA_8u_C1AC4R(dpFrameBayer[CurrentFrameID], nPitchBayer, osize, orect, dpFrame, nPitch,NPPI_BAYER_BGGR, NPPI_INTER_UNDEFINED,255); //NPPI_INTER_CUBIC, NPPI_INTER_UNDEFINED
        
        // Perform white balancing correction
        applyWhiteBalance(dpFrame, nPitch, VideoWidth, VideoHeight, GainR, GainG, GainB);

        // Update the texture in gaming engine
        UpdateTextureFromGPU();

        // Sleep for some time to wait for next update (artificial frame rate)
        std::this_thread::sleep_for(std::chrono::milliseconds(20));

    }// end of thread while loop


    printf("The camera acquisiton is closed successfully!\n");
    
    // Deallocate dynamic GPU memories
    nppiFree(dpFrame);
    for (int i = 0; i < 240; i++) {
        nppiFree(dpFrameBayer[i]);
    }

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



int AVideoAcquisition::applyWhiteBalance(Npp8u* img_d, int img_pitch, int width, int height, float _gain_r, float _gain_g, float _gain_b) {

    // white balance color twist
    Npp32f wbTwist[3][4] = {
        { 1.0, 0.0, 0.0, 0.0 },
        { 0.0, 1.0, 0.0, 0.0 },
        { 0.0, 0.0, 1.0, 0.0 }
    };
    wbTwist[0][0] = _gain_r;
    wbTwist[1][1] = _gain_g;
    wbTwist[2][2] = _gain_b;
    NppiSize osize;
    osize.width = width;
    osize.height = height;

    nppiColorTwist32f_8u_C4IR(img_d, img_pitch, osize, wbTwist);


    return 0;
}

