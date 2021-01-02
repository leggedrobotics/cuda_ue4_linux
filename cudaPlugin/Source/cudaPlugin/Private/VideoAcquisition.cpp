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
    memset(&xiImage, 0, sizeof(xiImage));
    xiImage.size = sizeof(XI_IMG);

    // Retrieving a handle to the camera device
    printf("Opening camera ...\n");
    xiOpenDeviceBy(XI_OPEN_BY_SN, TCHAR_TO_UTF8(*CameraSerial), &xiHandle);
    HandleResult(stat, "xiOpenDevice");

    // show serial number of this camera
    char sn[20] = "";
    xiGetParamString(xiHandle, XI_PRM_DEVICE_SN, sn, sizeof(sn));
    printf("Camera opened with Serial number: %s\n", sn);


    printf("Exposure is set to %d us\n", exposure);
    // Setting "exposure" parameter (10ms=10000us)
    stat = xiSetParamInt(xiHandle, XI_PRM_EXPOSURE, exposure);
    //HandleResult(stat,"xiSetParam (exposure set)");
    char tempType[200] = "";
    std::string devTypePCIe("PCIe");
    std::string devTypeUSB("U3V");

    xiGetParamString(xiHandle, XI_PRM_DEVICE_TYPE, &tempType, sizeof(tempType));


    if (devTypePCIe.compare(tempType) == 0) {

        printf("This is a PCIe camera--> GPUDirect is going to be enabled\n");
        //GPUDirect
        xiSetParamInt(xiHandle, XI_PRM_BUFFER_POLICY, XI_BP_UNSAFE);
        xiSetParamInt(xiHandle, XI_PRM_IMAGE_DATA_FORMAT, XI_FRM_TRANSPORT_DATA);
        xiSetParamInt(xiHandle, XI_PRM_OUTPUT_DATA_BIT_DEPTH, 8);
        xiSetParamInt(xiHandle, XI_PRM_TRANSPORT_DATA_TARGET, XI_TRANSPORT_DATA_TARGET_GPU_RAM); // or XI_TRANSPORT_DATA_TARGET_UNIFIED, XI_TRANSPORT_DATA_TARGET_ZEROCOPY, XI_TRANSPORT_DATA_TARGET_GPU_RAM

        int payload_size = 0;
        stat = xiGetParamInt(xiHandle, XI_PRM_IMAGE_PAYLOAD_SIZE, &payload_size);
        stat = xiSetParamInt(xiHandle, XI_PRM_ACQ_BUFFER_SIZE, payload_size * 4);

    }
    else if (devTypeUSB.compare(tempType) == 0) {

        printf("This is a USB 3.0 camera--> GPU Zero-Copy is going to be enabled\n");

        //GPU Zero-Copy
        xiSetParamInt(xiHandle, XI_PRM_BUFFER_POLICY, XI_BP_UNSAFE);
        xiSetParamInt(xiHandle, XI_PRM_IMAGE_DATA_FORMAT, XI_FRM_TRANSPORT_DATA);
        xiSetParamInt(xiHandle, XI_PRM_OUTPUT_DATA_BIT_DEPTH, 8);
        xiSetParamInt(xiHandle, XI_PRM_TRANSPORT_DATA_TARGET, XI_TRANSPORT_DATA_TARGET_ZEROCOPY); // or XI_TRANSPORT_DATA_TARGET_UNIFIED, XI_TRANSPORT_DATA_TARGET_ZEROCOPY, XI_TRANSPORT_DATA_TARGET_GPU_RAM

        xiSetParamInt(xiHandle, XI_PRM_ACQ_BUFFER_SIZE, 4628480);
        xiSetParamInt(xiHandle, XI_PRM_ACQ_BUFFER_SIZE_UNIT, 8);
        xiSetParamInt(xiHandle, XI_PRM_BUFFERS_QUEUE_SIZE, 8);

        // set data rate
        xiSetParamInt(xiHandle, XI_PRM_LIMIT_BANDWIDTH, 3170);
        // enable the limiting
        xiSetParamInt(xiHandle, XI_PRM_LIMIT_BANDWIDTH_MODE, XI_ON);

    }
    else {


        printf("The camera interface - %s - is not detected!\n", tempType);


    }

    // set acquisition to frame rate mode
    xiSetParamInt(xiHandle, XI_PRM_ACQ_TIMING_MODE, XI_ACQ_TIMING_MODE_FRAME_RATE_LIMIT);
    // set frame rate
    xiSetParamInt(xiHandle, XI_PRM_FRAMERATE, FPS);


}

void AVideoAcquisition::threadLoop()
{

    printf("Starting camera acquisition...\n");
    stat = xiStartAcquisition(xiHandle);
    HandleResult(stat, "xiStartAcquisition");

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

    while(!stopped) {

        stat = xiGetImage(xiHandle, 5000, &xiImage);

        if (stat == XI_OK) {

            // Perform Bayer to RGB conversion
            status = nppiCFAToRGBA_8u_C1AC4R((Npp8u*)xiImage.bp, VideoWidth, osize, orect, dpFrame, nPitch, NPPI_BAYER_BGGR, NPPI_INTER_UNDEFINED, 255); //NPPI_INTER_CUBIC, NPPI_INTER_UNDEFINED

            // Perform white balancing correction
            applyWhiteBalance(dpFrame, nPitch, VideoWidth, VideoHeight, GainR, GainG, GainB);

            // Update the texture in gaming engine
            UpdateTextureFromGPU();

        }

    }// end of thread while loop

    printf("Stopping camera acquisition...\n");
    xiStopAcquisition(xiHandle);
    xiCloseDevice(xiHandle);
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

