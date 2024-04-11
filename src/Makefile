#Makefile 
#define variables
objects= main.o timer.o #cpu_kernels.o scan/scan2/scan.o scan/scan2/kernels.o scan/scan2/utils.o 
#DRAG.o IOdata.o DRAG_kernels.o preprocessing.o preprocessing_kernels.o
#compare_cpu_gpu.o 
NVCC= nvcc              #cuda c compiler
opt= -use_fast_math -O2 -g -G           #optimization flag
#ARCH= -arch=sm_70        #cuda compute capability
#ARCH=-gencode=arch=compute_60,code=sm_60

MPI_HOME = /opt/mpi/openmpi-4.1.1-gcc8

LIBS=
MPIFLAGS = -I$(MPI_HOME)/include -L$(MPI_HOME)/lib -lmpi 
execname= main
L1CACHE_DISABLE = -Xptxas -dlcm=cg

#compile
$(execname): $(objects)
	#$(NVCC) $(opt) $(L1CACHE_DISABLE) -o $(execname) $(objects) $(LIBS)
	$(NVCC) $(MPIFLAGS) $(opt) --expt-relaxed-constexpr -o $(execname) $(objects) $(LIBS)

#DRAG.o: DRAG.cu
#	$(NVCC) $(opt) $(ARCH) -c DRAG.cu -D SEGMENT_LEN=$(SEGM_LEN)
#IOdata.o: IOdata.cpp
#	$(NVCC) $(opt) $(ARCH) -std=c++11 -c IOdata.cpp
#DRAG_kernels.o: DRAG_kernels.cu
#	$(NVCC) $(opt) $(ARCH) -c DRAG_kernels.cu -D SEGMENT_LEN=$(SEGM_LEN)
#preprocessing.o: preprocessing.cu
#	$(NVCC) $(opt) $(ARCH) -c preprocessing.cu -D SEGMENT_LEN=$(SEGM_LEN)
#preprocessing_kernels.o: preprocessing_kernels.cu
#	$(NVCC) $(opt) $(ARCH) -c preprocessing_kernels.cu -D SEGMENT_LEN=$(SEGM_LEN)
#cpu_kernels.o: cpu_kernels.cpp
#	$(NVCC) $(opt) $(ARCH) -std=c++11 -c cpu_kernels.cpp
timer.o: timer.cu
	$(NVCC) $(opt) $(ARCH) -c timer.cu
#main.o: main.cpp
#	$(NVCC) $(opt) $(ARCH) -std=c++11 -c main.cpp
main.o: main.cu
	$(NVCC) $(MPIFLAGS) $(opt) --expt-relaxed-constexpr $(ARCH) -std=c++11 -c main.cu 
#-D_DEBUG_

#clean Makefile
clean:
	rm $(objects)
#end of Makefile


##ARCH=-arch=sm_50 \ 
##-gencode=arch=compute_50,code=sm_50 \ 
##-gencode=arch=compute_52,code=sm_52 \ 
##-gencode=arch=compute_60,code=sm_60 \ 
##-gencode=arch=compute_61,code=sm_61 \ 
##-gencode=arch=compute_70,code=sm_70 \ 
##-gencode=arch=compute_75,code=sm_75 \
##-gencode=arch=compute_75,code=compute_75 
