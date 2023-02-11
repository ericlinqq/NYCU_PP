CXX := mpicxx
CXXFLAGS := -O3

all: matmul

matmul: main.o matmul.o
	$(CXX) $(CXXFLAGS) main.o matmul.o -o matmul

%.o: %.cc
	${CXX} ${CXXFLAGS} -c $^ -o $@

.PHONY: clean test
clean:
	rm -rf matmul *.o *~