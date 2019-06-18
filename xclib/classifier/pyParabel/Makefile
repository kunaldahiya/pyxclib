INC=-I../Tools/c++
CXXFLAGS=-std=c++11 -O3 -g
LIBFLAGS=-pthread -I/home/cse/msr/siy177545/Eigen -I/home/cse/msr/siy177545/scratch/anaconda3/include/python3.7m -I/home/cse/msr/siy177545/.local/include/python3.7m

all: clean parabel

parabel_train:
	$(CXX) -o parabel_train $(CXXFLAGS) $(INC) parabel_train.cpp parabel.cpp $(LIBFLAGS)

parabel_predict:
	$(CXX) -o parabel_predict $(CXXFLAGS) $(INC) parabel_predict.cpp parabel.cpp $(LIBFLAGS)

parabel:
	$(CXX) -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` parabel.cpp $(LIBFLAGS) -o parabel`python3-config --extension-suffix`
clean:
	rm -f parabel_train parabel_predict

