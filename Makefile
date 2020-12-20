all: ply.o raytrace.o main

ply.o: ply.cpp
	g++ -c ply.cpp -ggdb -lGL -lGLU -lglut -Wall

raytrace.o: raytrace.cu
	nvcc -c --ptxas-options=-v -arch sm_75 raytrace.cu -lcuda -g

raytrace.ptx: raytrace.cu
	nvcc -ptx -arch sm_75 raytrace.cu

main: ply.o raytrace.o main.cpp
	g++ main.cpp ply.o raytrace.o -o main -ggdb -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lGL -lGLU -lglut -Wall -std=c++11