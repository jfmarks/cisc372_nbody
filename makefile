FLAGS = -DDEBUG
LIBS = -lm
ALWAYS_REBUILD = makefile

NVCC = nvcc
CC = gcc

NVCC_FLAGS = -arch=sm_61
GCC_FLAGS = -std=c99

nbody: nbody.o compute.o
	$(NVCC) $(FLAGS) $^ -o $@ $(LIBS)

nbody.o: nbody.cu planets.h config.h vector.h $(ALWAYS_REBUILD)
	$(NVCC) $(FLAGS) $(NVCC_FLAGS) -c $<

compute.o: compute.cu config.h vector.h $(ALWAYS_REBUILD)
	$(NVCC) $(FLAGS) $(NVCC_FLAGS) -c $<

clean:
	rm -f *.o nbody
