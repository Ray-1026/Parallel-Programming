default: conv

CC = gcc
FLAGS = -O3 -DCL_TARGET_OPENCL_VERSION=220 -lOpenCL -m64 -ffloat-store -w -g

# cuda
COMPUTE_CAPABILITY ?= 61
NVCC = nvcc
CUDA_LINK_FLAGS =  -rdc=true -gencode=arch=compute_$(COMPUTE_CAPABILITY),code=sm_$(COMPUTE_CAPABILITY) -Xcompiler '-fPIC' -lOpenCL
CUDA_COMPILE_FLAGS = --device-c -gencode=arch=compute_$(COMPUTE_CAPABILITY),code=sm_$(COMPUTE_CAPABILITY) -Xcompiler '-fPIC' -g -O3

OBJS = main.o bmpfuncs.o hostFE.o serialConv.o helper.o

conv: $(OBJS)
ifdef CUDA
	$(NVCC) $(CUDA_LINK_FLAGS) -o $@ $(OBJS)
else
	$(CC) -o $@ $(OBJS) $(FLAGS)
endif

%.o: %.cu
ifdef CUDA
	$(NVCC) $(CUDA_COMPILE_FLAGS) -c $< -o $@
endif	

%.o: %.c
	$(CC) -c $(FLAGS) $< -o $@

clean:
	$(RM) conv *.o output.bmp ref.bmp