// Fill out your copyright notice in the Description page of Project Settings.
#pragma once



#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "GameFramework/Actor.h"
#include "Engine/TextureRenderTarget2D.h"

#include "cuda.h"
// includes, cuda
#include <cuda_runtime.h>
#include "npp.h"




#include "Windows/AllowWindowsPlatformTypes.h"
#include "cuda_d3d11_interop.h"
#include "d3d11.h"
#include "Windows/HideWindowsPlatformTypes.h"

#include <memory.h>
#include <string>
#include <iostream>
#include <thread>
#include <vector>

#include <fstream>      // std::ifstream, std::ofstream

/*
#include "DynamicRHI.h"
#include "RenderResource.h"
#include "RHICommandList.h"
#include "Materials/MaterialInstanceDynamic.h"
*/

#include "Windows/AllowWindowsPlatformTypes.h"
#include "cuda.h"
#include "cuda_d3d11_interop.h"
#include "d3d11.h"
#include "Windows/HideWindowsPlatformTypes.h"

#include "DynamicRHI.h"
#include "D3D11RHI.h"
#include "D3D11RHIBasePrivate.h"
#include "D3D11StateCachePrivate.h"
#include "D3D11Util.h"
#include "D3D11State.h"
#include "D3D11Resources.h"
#include "Materials/MaterialInstanceDynamic.h"

#include "RenderResource.h"
#include "RHICommandList.h"


#include <iostream>
#include <chrono>
#include <thread>


#ifndef _WIN32_WINNT		// Allow use of features specific to Windows XP or later.                   
#define _WIN32_WINNT 0x0501	// Change this to the appropriate value to target other versions of Windows.
#endif	

#include <tchar.h>
#include <windows.h>
#include <conio.h>
#include <process.h>

#include "xiApi.h"
#include <memory.h>

#include "VideoAcquisition.generated.h"


#define HandleResult(res,place) if (res!=XI_OK) {printf("Error after %s (%d)\n",place,res);}



UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class CUDAPLUGIN_API AVideoAcquisition : public AActor
{
	GENERATED_BODY()

public:
	// Sets default values for this component's properties
	AVideoAcquisition();
	~AVideoAcquisition();

	//The texture where the Video is rendered to
	UPROPERTY(EditAnywhere)
		UTextureRenderTarget2D* RenderTarget;
	
	//Video configuration
	UPROPERTY(EditAnywhere)
		int VideoWidth = 2064;
	UPROPERTY(EditAnywhere)
		int VideoHeight = 1544;
	UPROPERTY(EditAnywhere)
		int FPS = 50;
	UPROPERTY(EditAnywhere)
		int exposure = 10000;
	UPROPERTY(EditAnywhere)
		FString CameraSerial = "CECAU1944005";
	UPROPERTY(EditAnywhere)
		float GainR = 0.7;
	UPROPERTY(EditAnywhere)
		float GainG = 0.6;
	UPROPERTY(EditAnywhere)
		float GainB = 1.8;

	virtual void Tick(float DeltaTime) override;

    bool stopped;

	bool InitCUDA();
    void InitCamera();
    void threadLoop();
    void stop() { stopped = true; }
    void start() { stopped = false; }

    void UpdateTextureFromGPU();
	int applyWhiteBalance(Npp8u* img_d, int img_pitch, int width, int height, float _gain_r, float _gain_g, float _gain_b);

protected:
	// Called when the game starts
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type) override;

private:
	CUcontext cuContext = NULL;

	cudaGraphicsResource* CudaResource;
	cudaGraphicsResource* Resources[1];

	int DISPLAY_WIDTH;
	int DISPLAY_HEIGHT;

	bool GameEnded = false;

    uint8_t *dpFrameBayer[240];
    int nPitchBayer;
    uint8_t *dpFrame;
	int nPitch = 0;

    uint8_t *destPtr;
    uint32 anydummy;


	// Ximea API variables
	HANDLE xiHandle = NULL;
	XI_RETURN stat = XI_OK;

	// image buffer
	XI_IMG xiImage;

    std::thread *cameraThread = nullptr;

    

};
