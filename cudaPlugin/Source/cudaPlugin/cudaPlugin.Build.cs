// Copyright 1998-2019 Epic Games, Inc. All Rights Reserved.
using System;
using System.IO;
using UnrealBuildTool;

public class cudaPlugin : ModuleRules
{
	public cudaPlugin(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;
					
		
			PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				"RHI",
				"RenderCore",
				"VulkanRHI",
				"Projects"

			}
			);
			
		
		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
				"CoreUObject",
				"Engine",
				"Slate",
				"SlateCore",
				"BuildSettings"
				// ... add private dependencies that you statically link with here ...	
			}
			);
		


		//Include RHI Headers for texture access in GPU memory
           	string EnginePath = Path.GetFullPath(Target.RelativeEnginePath);
		PrivateIncludePaths.AddRange(
			new string[] {
				// ... add other private include paths required here ...

				EnginePath + "Source/Runtime/VulkanRHI/Private/",
                		EnginePath + "Source/Runtime/VulkanRHI/Private/Linux/",
				EnginePath + "Source/ThirdParty/Vulkan/Include/vulkan"

			}
			);

		PublicIncludePaths.AddRange(
		new string[]
		{
		EnginePath + "Source/Runtime/VulkanRHI/Public/",
		}
		);


		//Include CUDA
		string cuda_path = "/usr/local/cuda";
		string cuda_include = "include";
		string cuda_lib = "lib64";
		string cuda_lib_stubs = "lib64/stubs";

		PublicIncludePaths.Add(Path.Combine(cuda_path, cuda_include));

		PublicAdditionalLibraries.Add(Path.Combine(cuda_path, cuda_lib, "libcudart_static.a"));
		PublicAdditionalLibraries.Add(Path.Combine(cuda_path, cuda_lib_stubs, "libcuda.so"));
		PublicAdditionalLibraries.Add(Path.Combine(cuda_path, cuda_lib, "libnppial.so"));
		PublicAdditionalLibraries.Add(Path.Combine(cuda_path, cuda_lib, "libnppicc.so"));
		PublicAdditionalLibraries.Add(Path.Combine(cuda_path, cuda_lib, "libnppig.so"));
		PublicAdditionalLibraries.Add(Path.Combine(cuda_path, cuda_lib, "libnppial.so"));
		PublicAdditionalLibraries.Add(Path.Combine(cuda_path, cuda_lib, "libnppif.so"));
		PublicAdditionalLibraries.Add(Path.Combine(cuda_path, cuda_lib, "libnppist.so"));
		PublicAdditionalLibraries.Add(Path.Combine(cuda_path, cuda_lib, "libnppidei.so"));
		PublicAdditionalLibraries.Add(Path.Combine(cuda_path, cuda_lib, "libnpps.so"));
		PublicAdditionalLibraries.Add(Path.Combine(cuda_path, cuda_lib, "libnppisu.so"));

		
		DynamicallyLoadedModuleNames.AddRange(
			new string[]
			{
				// ... add any modules that your module loads dynamically here ...
			}
			);
	}
}
