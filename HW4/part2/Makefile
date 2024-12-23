CXX := mpicxx
CXXFLAGS := -O3 -Wall -march=native -ffast-math

OBJECTS := matmul

all: 
	$(CXX) $(CXXFLAGS) -o $(OBJECTS) main.cc matmul.cc

test: all
	srun --mpi=pmix -N $(ARG1) matmul $(ARG2) > output.txt
	tail -n 1 output.txt
	rm -rf output.txt

.PHONY: clean
clean:
	$(RM) $(OBJECTS)