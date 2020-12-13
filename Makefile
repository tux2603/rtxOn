CFLAGS = -ggdb -lGL -lGLU -lglut -Wall -ansi -pedantic -Wno-unused-parameter -Wno-unused-but-set-variable -Wno-unused-variable -pthread

all: ply.o rayTracing

ply.o: ply.cpp
	g++ -c ply.cpp $(CFLAGS)

rayTracing: ply.o main.cpp
	g++ main.cpp ply.o -o main $(CFLAGS) -std=c++11