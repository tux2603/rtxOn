CFLAGS = -ggdb -lGL -lGLU -lglut -Wall -ansi -pedantic -Wno-unused-parameter -Wno-unused-but-set-variable -Wno-unused-variable -pthread -O3

all: utils.o rayTracing

utils.o: utils.cpp
	g++ -c utils.cpp $(CFLAGS) -std=c++11

mathTest: utils.o mathTest.cpp
	g++ mathTest.cpp utils.o -o mathTest $(CFLAGS) -std=c++11

rayTracing: utils.o plyfile.cpp rayTracing.cpp
	g++ -c plyfile.cpp $(CFLAGS)
	g++ rayTracing.cpp plyfile.o utils.o -o rayTracing $(CFLAGS) -std=c++11