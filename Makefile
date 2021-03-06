CC = g++
CFLAGS = -O3 -std=c++11 -MMD

CEC_DIR := ./CEC2013_niching_benchmark
CEC_SRC_FILES := $(wildcard $(CEC_DIR)/*.cpp)
CEC_OBJ_FILES := $(patsubst $(CEC_DIR)/%.cpp,$(CEC_DIR)/%.o,$(CEC_SRC_FILES))
CEC_DEP_FILES := $(patsubst $(CEC_DIR)/%.cpp,$(CEC_DIR)/%.d,$(CEC_SRC_FILES))

HVEA_DIR := ./HillVallEA
HVEA_SRC_FILES := $(wildcard $(HVEA_DIR)/*.cpp)
HVEA_OBJ_FILES := $(patsubst $(HVEA_DIR)/%.cpp,$(HVEA_DIR)/%.o,$(HVEA_SRC_FILES))
HVEA_DEP_FILES := $(patsubst $(HVEA_DIR)/%.cpp,$(HVEA_DIR)/%.d,$(HVEA_SRC_FILES))

all: example_cec2013_benchmark example_simple multimodal_AIY multimodal_AFD multimodal_RIM BenchmarkLinearCatKLt BenchmarkLinearCapKirKLt BenchmarkCubicCapKirKLt

example_cec2013_benchmark: example_CEC2013_benchmark.o $(HVEA_OBJ_FILES) $(CEC_OBJ_FILES)
	$(CC) $(CFLAGS) -o $@ example_CEC2013_benchmark.o $(HVEA_OBJ_FILES) $(CEC_OBJ_FILES)

example_simple: example_simple.o $(HVEA_OBJ_FILES)
	$(CC) $(CFLAGS) -o $@ example_simple.o $(HVEA_OBJ_FILES)

multimodal_AIY: multimodal_AIY.o $(HVEA_OBJ_FILES)
	$(CC) $(CFLAGS) -o $@ multimodal_AIY.o $(HVEA_OBJ_FILES)

multimodal_AFD: multimodal_AFD.o $(HVEA_OBJ_FILES)
	$(CC) $(CFLAGS) -o $@ multimodal_AFD.o $(HVEA_OBJ_FILES)

multimodal_RIM: multimodal_RIM.o $(HVEA_OBJ_FILES)
	$(CC) $(CFLAGS) -o $@ multimodal_RIM.o $(HVEA_OBJ_FILES)

BenchmarkLinearCatKLt: BenchmarkLinearCatKLt.o $(HVEA_OBJ_FILES)
	$(CC) $(CFLAGS) -o $@ BenchmarkLinearCatKLt.o $(HVEA_OBJ_FILES)

BenchmarkLinearCapKirKLt: BenchmarkLinearCapKirKLt.o $(HVEA_OBJ_FILES)
	$(CC) $(CFLAGS) -o $@ BenchmarkLinearCapKirKLt.o $(HVEA_OBJ_FILES)

BenchmarkCubicCapKirKLt: BenchmarkCubicCapKirKLt.o $(HVEA_OBJ_FILES)
	$(CC) $(CFLAGS) -o $@ BenchmarkCubicCapKirKLt.o $(HVEA_OBJ_FILES)


%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(CEC_OBJ_FILES) $(CEC_DEP_FILES) $(HVEA_OBJ_FILES) $(HVEA_DEP_FILES) *.d *.o

clean_run:
	rm -f example_cec2013_benchmark example_simple multimodal_AIY multimodal_AFD multimodal_RIM BenchmarkLinearCatKLt BenchmarkLinearCapKirKLt BenchmarkCubicCapKirKLt elites*.dat statistics*.dat 
