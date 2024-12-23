CXX := g++
CXXFLAGS := -O3 -std=c++17 -Wall -pthread -march=native -funroll-loops -fno-rtti -mtune=native -fno-exceptions -mavx2 -ffast-math

OBJECTS := pi.out

all: clean
	$(CXX) $(CXXFLAGS) -Iinclude -o $(OBJECTS) pi.cpp

clean:
	$(RM) *.out

run: all
	srun -c 4 bash -c "time (./$(OBJECTS) 3 100000000; ./$(OBJECTS) 4 100000000)"