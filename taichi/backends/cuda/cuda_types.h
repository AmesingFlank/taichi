#pragma once

using CUexternalMemory_frank = void*;


// copied from <cuda.h>

typedef enum class CUexternalMemoryHandleType_enum_frank {
    /**
     * Handle is an opaque file descriptor
     */
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD          = 1,
    /**
     * Handle is an opaque shared NT handle
     */
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32       = 2,
    /**
     * Handle is an opaque, globally shared handle
     */
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT   = 3,
    /**
     * Handle is a D3D12 heap object
     */
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP         = 4,
    /**
     * Handle is a D3D12 committed resource
     */
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE     = 5,
    /**
     * Handle is a shared NT handle to a D3D11 resource
     */
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE     = 6,
    /**
     * Handle is a globally shared handle to a D3D11 resource
     */
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT = 7,
    /**
     * Handle is an NvSciBuf object
     */
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF = 8
} CUexternalMemoryHandleType_frank;

typedef struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_frank {
    /**
     * Type of the handle
     */
    CUexternalMemoryHandleType_frank type;
    union {
        /**
         * File descriptor referencing the memory object. Valid
         * when type is
         * ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD
         */
        int fd;
        /**
         * Win32 handle referencing the semaphore object. Valid when
         * type is one of the following:
         * - ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32
         * - ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT
         * - ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP
         * - ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE
         * - ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE
         * - ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT
         * Exactly one of 'handle' and 'name' must be non-NULL. If
         * type is one of the following:
         * ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT
         * ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT
         * then 'name' must be NULL.
         */
        struct {
            /**
             * Valid NT handle. Must be NULL if 'name' is non-NULL
             */
            void *handle;
            /**
             * Name of a valid memory object.
             * Must be NULL if 'handle' is non-NULL.
             */
            const void *name;
        } win32;
        /**
         * A handle representing an NvSciBuf Object. Valid when type
         * is ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF
         */
        const void *nvSciBufObject;
    } handle;
    /**
     * Size of the memory allocation
     */
    unsigned long long size;
    /**
     * Flags must either be zero or ::CUDA_EXTERNAL_MEMORY_DEDICATED
     */
    unsigned int flags;
    unsigned int reserved[16];
} CUDA_EXTERNAL_MEMORY_HANDLE_DESC_frank;