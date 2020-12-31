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

#include <memory.h>
#include <string>
#include <iostream>
#include <thread>
#include <vector>

#include <fstream>      // std::ifstream, std::ofstream

#include "DynamicRHI.h"
#include "RenderResource.h"
#include "RHICommandList.h"
#include "Materials/MaterialInstanceDynamic.h"
//#include "VulkanLinuxPlatform.h"
//#include "VulkanPlatformDefines.h"
/*
#include "VulkanState.h"
#include "VulkanUtil.h"
#include "VulkanResources.h"
*/

#include "VulkanRHIPrivate.h"
#include "VulkanPendingState.h"
#include "VulkanContext.h"
#include "EngineGlobals.h"
#include "VulkanLLM.h"
#include "VulkanResources.h"


#include "VulkanCommon.h"
#include "VulkanConfiguration.h"
#include "VulkanDynamicRHI.h"
#include "VulkanGlobals.h"
#include "VulkanMemory.h"
#include "VulkanRHIBridge.h"
#include "VulkanShaderResources.h"
#include "VulkanUtil.h"


#include "BoundShaderStateCache.h"
#include "VulkanShaderResources.h"
#include "VulkanState.h"
#include "Misc/ScopeRWLock.h"



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

	void Acquire();

	virtual void Tick(float DeltaTime) override;

    bool stopped;

    void InitCamera();
    void threadLoop();
    void stop() { stopped = true; }
    void start() { stopped = false; }

    void UpdateTextureFromGPU();


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


    std::thread *cameraThread = nullptr;

    uint8_t vkDeviceUUID[VK_UUID_SIZE];

};
