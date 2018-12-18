CC=g++
CFLAGS= -Wall
LIB = -lm 
OUTPUT = -o bin/a.out
SOURCES = StaticMatrixTest.cc StaticMatrix.cc TestingUtils.cc

all: 
	$(CC) $(SOURCES) $(CFLAGS) $(OUTPUT) $(LIB)
