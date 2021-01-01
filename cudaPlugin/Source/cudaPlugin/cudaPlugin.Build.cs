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
				"D3D11RHI",
				"Projects"
		// ... add other public dependencies that you statically link with here ...
		}
		);



		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
				"CoreUObject",
				"Engine",
				"Slate",
				"SlateCore",
				// ... add private dependencies that you statically link with here ...	
			}
			);




		//Include RHI Headers for texture access in GPU memory
		string EnginePath = Path.GetFullPath(Target.RelativeEnginePath);
		PrivateIncludePaths.AddRange(
			new string[]
			{
				EnginePath + "Source/Runtime/Windows/D3D11RHI/Private/",
				EnginePath + "Source/Runtime/Windows/D3D11RHI/Private/Windows/"
			}
		);

		PublicIncludePaths.AddRange(
			new string[]
			{
				EnginePath + "Source/Runtime/Windows/D3D11RHI/Public/"
			}
		);

		//Include CUDA
		string cuda_path = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2";
		string cuda_include = "include";
		string cuda_lib = "lib/x64";

		PublicIncludePaths.Add(Path.Combine(cuda_path, cuda_include));

		//PublicAdditionalLibraries.Add(Path.Combine(cuda_path, cuda_lib, "cudart.lib"));
		PublicAdditionalLibraries.Add(Path.Combine(cuda_path, cuda_lib, "cudart_static.lib"));
		PublicAdditionalLibraries.Add(Path.Combine(cuda_path, cuda_lib, "cuda.lib"));
		PublicAdditionalLibraries.Add(Path.Combine(cuda_path, cuda_lib, "nppial.lib"));
		PublicAdditionalLibraries.Add(Path.Combine(cuda_path, cuda_lib, "nppig.lib"));
		PublicAdditionalLibraries.Add(Path.Combine(cuda_path, cuda_lib, "nppial.lib"));
		PublicAdditionalLibraries.Add(Path.Combine(cuda_path, cuda_lib, "nppif.lib"));
		PublicAdditionalLibraries.Add(Path.Combine(cuda_path, cuda_lib, "nppist.lib"));
		PublicAdditionalLibraries.Add(Path.Combine(cuda_path, cuda_lib, "nppidei.lib"));
		PublicAdditionalLibraries.Add(Path.Combine(cuda_path, cuda_lib, "npps.lib"));
		PublicAdditionalLibraries.Add(Path.Combine(cuda_path, cuda_lib, "nppisu.lib"));
		PublicAdditionalLibraries.Add(Path.Combine(cuda_path, cuda_lib, "nppicc.lib"));
		PublicAdditionalLibraries.Add(Path.Combine(cuda_path, cuda_lib, "nppig.lib"));
		
		
	



	}
}
