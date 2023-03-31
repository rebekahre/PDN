
NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = convolution_CUDA
OBJ	        = convolution_CUDA.o

default: $(EXE)

run:
	make 
	./convolution_CUDA 2048 2048 /home/lee0305/test_data/Problem_3/mat_input.csv mat_output.csv time.csv


convolution_CUDA.o: convolution_CUDA.cu kernel.cu 
	$(NVCC) -c -o $@ convolution_CUDA.cu $(NVCC_FLAGS)


$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
