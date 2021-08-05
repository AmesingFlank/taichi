#pragma once

#include <driver_types.h>
#include <cuda.h>
#include <vector>
#include "../common/constants.h"
//#include "taichi/backends/cuda/cuda_driver.h"


inline static void handle_error(CUresult err, const char* file, int line) {
	if (err != CUDA_SUCCESS) {
        const char* msg;
        cuGetErrorString(err,&msg);
		printf("CUDA error: %s in %s at line %d\n", msg, file, line);
		exit(EXIT_FAILURE);
	}
}
inline void get_last_cuda_error(const char *errorMessage, const char *file, const int line)
{
    cuCtxSynchronize();
    // cudaError_t err = cudaGetLastError();

    // if (cudaSuccess != err)
    // {
    //     fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
    //             file, line, errorMessage, (int)err, cudaGetErrorString(err));
    //     cudaDeviceReset();
    //     exit(EXIT_FAILURE);
    // }
}



#if GGUI_DEBUG
// This will output the proper error string when calling cudaGetLastError
#define CHECK_CUDA_ERROR(msg) get_last_cuda_error (msg, __FILE__, __LINE__)

#define HANDLE_ERROR( err ) \
    cuCtxSynchronize();\
    handle_error( err, __FILE__, __LINE__ )

#else

#define CHECK_CUDA_ERROR(msg) 

#define HANDLE_ERROR( err ) err

#endif //GGUI_DEBUG


