#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

struct cudaUUID_t {
    char bytes[16];
};

struct cudaDeviceProp {
    char         name[256];
    struct cudaUUID_t uuid;
    char         luid[8];
    unsigned int luidDeviceNodeMask;
    size_t       totalGlobalMem;
    size_t       sharedMemPerBlock;
    int          regsPerBlock;
    int          warpSize;
    size_t       memPitch;
    int          maxThreadsPerBlock;
    int          maxThreadsDim[3];
    int          maxGridSize[3];
    int          clockRate;
    size_t       totalConstMem;
    int          major;
    int          minor;
};

static void* get_symbol(const char* libname, const char* symbol) {
    void* handle = dlopen(libname, RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) {
        if (libname[0] == 'l') {
            handle = dlopen("libcuda.so", RTLD_LAZY | RTLD_GLOBAL);
        }
    }
    if (!handle) {
        return dlsym(RTLD_NEXT, symbol);
    }
    return dlsym(handle, symbol);
}

// 1. Intercept cudaGetDeviceProperties
typedef int (*orig_cudaGetDeviceProperties_t)(struct cudaDeviceProp*, int);
int cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
    static orig_cudaGetDeviceProperties_t orig = NULL;
    if (!orig) {
        orig = (orig_cudaGetDeviceProperties_t)get_symbol("libcudart.so.12", "cudaGetDeviceProperties");
    }
    if (!orig) return 999;
    int err = orig(prop, device);
    if (err == 0) {
        prop->major = 12;
        prop->minor = 0;
    }
    return err;
}

// 2. Intercept cudaGetDeviceProperties_v2 (CUDA 12 v2 symbol)
int cudaGetDeviceProperties_v2(struct cudaDeviceProp *prop, int device) {
    static orig_cudaGetDeviceProperties_t orig = NULL;
    if (!orig) {
        orig = (orig_cudaGetDeviceProperties_t)get_symbol("libcudart.so.12", "cudaGetDeviceProperties_v2");
    }
    if (!orig) return 999;
    int err = orig(prop, device);
    if (err == 0) {
        prop->major = 12;
        prop->minor = 0;
    }
    return err;
}

// 3. Intercept cudaDeviceGetAttribute
typedef int (*orig_cudaDeviceGetAttribute_t)(int*, int, int);
int cudaDeviceGetAttribute(int *value, int attr, int device) {
    static orig_cudaDeviceGetAttribute_t orig = NULL;
    if (!orig) {
        orig = (orig_cudaDeviceGetAttribute_t)get_symbol("libcudart.so.12", "cudaDeviceGetAttribute");
    }
    if (!orig) return 999;
    int err = orig(value, attr, device);
    if (err == 0) {
        if (attr == 75) {
            *value = 12;
        } else if (attr == 76) {
            *value = 0;
        }
    }
    return err;
}

// 4. Intercept cuDeviceGetAttribute
typedef int (*orig_cuDeviceGetAttribute_t)(int*, int, int);
int cuDeviceGetAttribute(int *pi, int attrib, int dev) {
    static orig_cuDeviceGetAttribute_t orig = NULL;
    if (!orig) {
        orig = (orig_cuDeviceGetAttribute_t)get_symbol("libcuda.so.1", "cuDeviceGetAttribute");
    }
    if (!orig) return 999;
    int err = orig(pi, attrib, dev);
    if (err == 0) {
        if (attrib == 75) { // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
            *pi = 12;
        } else if (attrib == 76) { // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
            *pi = 0;
        }
    }
    return err;
}

// 5. Intercept cuDeviceComputeCapability
typedef int (*orig_cuDeviceComputeCapability_t)(int*, int*, int);
int cuDeviceComputeCapability(int *major, int *minor, int dev) {
    static orig_cuDeviceComputeCapability_t orig = NULL;
    if (!orig) {
        orig = (orig_cuDeviceComputeCapability_t)get_symbol("libcuda.so.1", "cuDeviceComputeCapability");
    }
    if (!orig) return 999;
    int err = orig(major, minor, dev);
    if (err == 0) {
        *major = 12;
        *minor = 0;
    }
    return err;
}

// 6. Intercept nvmlDeviceGetCudaComputeCapability
typedef int (*orig_nvmlDeviceGetCudaComputeCapability_t)(void*, int*, int*);
int nvmlDeviceGetCudaComputeCapability(void* device, int *major, int *minor) {
    static orig_nvmlDeviceGetCudaComputeCapability_t orig = NULL;
    if (!orig) {
        orig = (orig_nvmlDeviceGetCudaComputeCapability_t)get_symbol("libnvidia-ml.so.1", "nvmlDeviceGetCudaComputeCapability");
    }
    if (!orig) return 999;
    int err = orig(device, major, minor);
    if (err == 0) {
        *major = 12;
        *minor = 0;
    }
    return err;
}
