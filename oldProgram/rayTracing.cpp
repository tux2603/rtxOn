#include <GL/freeglut.h>
#include <GL/gl.h>
#include <GL/glut.h>
#include <algorithm>
#include <bits/types/FILE.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <math.h>
#include <pthread.h>
#include <string.h>

#include "ply.h"
#include "utils.h"

// #define DEBUG
// #define STIPPLE
// #define FACE_COLORS

// Set this to -1 to draw all faces
#define MAX_FACES -1

#define SCREEN_WIDTH 256
#define SCREEN_HEIGHT 256

#define FUZZ_FACTOR 1e-6f

#define NUM_THREADS 16

#define FAR_PLANE_DISTANCE 1e3f

#define IMAGE_PLANE_DISTANCE 4.f

#define MAP(x, srcLow, srcHigh, destLow, destHigh) ((x) - (srcLow)) * ((destHigh) - (destLow)) / ((srcHigh) - (srcLow)) + (destLow)
#define MAX(a, b) ((a) > (b) ? (a) : (b))

typedef struct ThreadParams {
    int    firstRow;
    int    numRows;
    Vector cameraLocation;
    ThreadParams() {}
    ThreadParams(int firstRow, int numRows, Vector cameraLocation) : firstRow(firstRow), numRows(numRows), cameraLocation(cameraLocation) {}
} ThreadParams;



void  init(char *modelName);
void  draw();
void *rayTrace(void *args);

void keyDown(unsigned char key, int x, int y);
void keyUp(unsigned char key, int x, int y);
void mouseClick(int button, int state, int x, int y);
void mouseMove(int x, int y);

Triangle **triangles;
int        numTriangles;
GLfloat *  pixelBuffer;

float  imagePlaneDistance = 4.f;
Matrix cameraTranslation  = Matrix::translation(0, 0, 10);
Matrix cameraDirection    = Matrix::identity();

Vector i_a       = Vector(0, 0, 1);
Vector i_i       = Vector(0, 1, 0);
Vector i_s       = Vector(1, 1, 0);
float  k_d       = 0.3f;
float  k_s       = 0.4f;
float  k_a       = 0.3f;
float  shinyness = 50.f;

int      numFaces = 0, numVertices = 0, num_elems = 0;
Vertex **vlist = nullptr;

Face **flist = nullptr;

Vector rays[SCREEN_HEIGHT][SCREEN_WIDTH];

bool keyStates[256];
bool lastKeyStates[256];

bool mouseDown;
int  mouseX = 0, mouseY = 0;
int  lastMouseX = 0, lastMouseY = 0;

int main(int argc, char **argv) {
    char *modelFileName = (argc > 1 ? argv[1] : (char *)"ply/wrightFlyer.ply");

    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);

    glutInitWindowPosition(50, 100);
    glutInitWindowSize(SCREEN_WIDTH, SCREEN_HEIGHT);

    // Set the title of the window to show the file being displayed
    char *titleBuffer = (char *)myalloc(sizeof(char) * (strlen("CEG4500 project 1 | ") + strlen(modelFileName) + 1));
    sprintf(titleBuffer, "CEG4500 Project 4 | %s", modelFileName);
    glutCreateWindow(titleBuffer);

    init(modelFileName);

    // // Set up the mouse handling functions
    // glutMouseFunc(handleMouseClick);
    // glutMotionFunc(handleMouseMovement);

    glutDisplayFunc(draw);
    glutIdleFunc(glutPostRedisplay);

    glutKeyboardFunc(keyDown);
    glutKeyboardUpFunc(keyUp);
    glutMouseFunc(mouseClick);
    glutMotionFunc(mouseMove);

    glutMainLoop();

    free(titleBuffer);
}

void init(char *modelName) {

    if (SCREEN_HEIGHT % NUM_THREADS != 0) {
        std::cout << "Error: screen height must be a multiple of thread count" << std::endl;
        exit(1);
    }

    glClearColor(0, 0, 0, 1);

    pixelBuffer = new GLfloat[SCREEN_WIDTH * SCREEN_HEIGHT * 3];



    read_test(modelName, &numFaces, &numVertices, &vlist, &flist);

    // Calculate the center of the figure along each axis for centering
    float minX = 1e42,
          minY = 1e42,
          minZ = 1e42,
          maxX = -1e42,
          maxY = -1e42,
          maxZ = -1e42;

    for (int i = 0; i < numVertices; ++i) {
        for (int v = 0; v < flist[i]->nverts; ++v) {
            Vertex *vert = vlist[flist[i]->verts[v]];

            minX = std::min(minX, vert->x);
            minY = std::min(minY, vert->y);
            minZ = std::min(minZ, vert->z);


            maxX = std::max(maxX, vert->x);
            maxY = std::max(maxY, vert->y);
            maxZ = std::max(maxZ, vert->z);
        }
    }

    float deltaX = maxX - minX;
    float deltaY = maxY - minY;
    float deltaZ = maxZ - minZ;

    float maxDelta = MAX(deltaX, MAX(deltaY, deltaZ));

    Matrix transformations = Matrix::scale(1 / maxDelta, 1 / maxDelta, 1 / maxDelta);
    transformations *= Matrix::translation(-(maxX + minX) / 2, -(maxY + minY) / 2, -(maxZ + minZ) / 2);
    // transformations *=  Matrix::scale(1 / maxDelta, 1 / maxDelta, 1 / maxDelta);

#if (defined(MAX_FACES) && MAX_FACES > -1)
    numFaces = numFaces > MAX_FACES ? MAX_FACES : numFaces;
#endif

    // Compute the number of triangles in the mesh and allocate enough space to store them
    numTriangles = 0;
    for (int i = 0; i < numFaces; ++i)
        numTriangles += flist[i]->nverts - 2;
    triangles = new Triangle *[numTriangles];


    // Load in and initialize all of the triangles for faster reference
    for (int i = 0, triangleIdx = 0; i < numFaces; ++i) {
        Face *f = flist[i];
        for (int j = 1; j < f->nverts - 1; ++j, ++triangleIdx) {
            Vector v1 = transformations * Vector(vlist[f->verts[0]]);
            Vector v2 = transformations * Vector(vlist[f->verts[j]]);
            Vector v3 = transformations * Vector(vlist[f->verts[j + 1]]);

            triangles[triangleIdx] = new Triangle(new Vector(v1), new Vector(v2), new Vector(v3));
        }
    }

    // Precompute all the rays
    for (int y = 0; y < SCREEN_HEIGHT; ++y) {
        for (int x = 0; x < SCREEN_WIDTH; ++x) {
            rays[y][x] = Vector(MAP(x, 0, SCREEN_WIDTH - 1, -1.f, 1.f), MAP(y, 0, SCREEN_HEIGHT - 1, -1.f, 1.f), -imagePlaneDistance).normalize();
        }
    }
}

void keyDown(unsigned char key, int x, int y) {
#ifdef DEBUG
    printf("Key '%c' was pressed\n", key);
#endif
    keyStates[key] = true;
}

void keyUp(unsigned char key, int x, int y) {
#ifdef DEBUG
    printf("Key '%c' was released\n", key);
#endif
    keyStates[key] = false;
}

void mouseClick(int button, int state, int x, int y) {
#ifdef DEBUG
    printf("Mouse %d detected\n", button);
#endif
    if (button == GLUT_LEFT_BUTTON) {
        mouseDown  = state == GLUT_DOWN;
        mouseX     = x;
        mouseY     = y;
        lastMouseX = x;
        lastMouseY = y;
    } else if (button == 3)
        cameraTranslation *= Matrix::translation(0, 0, -0.1);
    else if (button == 4)
        cameraTranslation *= Matrix::translation(0, 0, 0.1);
}

void mouseMove(int x, int y) {
    mouseX = x;
    mouseY = y;
}

void draw() {

    double start = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now().time_since_epoch()).count();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    pthread_t *   threads = new pthread_t[NUM_THREADS];
    ThreadParams *params  = new ThreadParams[NUM_THREADS];

    Vector cameraLocation = cameraDirection * cameraTranslation * Vector();


#if (NUM_THREADS != 1)
    for (int i = 0; i < NUM_THREADS; ++i) {
        params[i] = ThreadParams(i * SCREEN_HEIGHT / NUM_THREADS, SCREEN_HEIGHT / NUM_THREADS, cameraLocation);
        pthread_create(&threads[i], NULL, &rayTrace, params + i);
    }

    for (int i = 0; i < NUM_THREADS; ++i) {
        pthread_join(threads[i], NULL);
    }
#else
    ThreadParams tp(0, SCREEN_HEIGHT, cameraLocation);
    rayTrace((void *)&tp);
#endif

    glDrawPixels(SCREEN_WIDTH, SCREEN_HEIGHT, GL_RGB, GL_FLOAT, (GLvoid *)pixelBuffer);

    glFlush();
    glutSwapBuffers();

    // imagePlaneDistance += 0.05f;

    cameraDirection *= Matrix::rotateY((lastMouseX - mouseX) * 0.01f) * Matrix::rotateX((lastMouseY - mouseY) * 0.01f);

    if (keyStates['a'] || keyStates['A']) {
        cameraTranslation *= Matrix::translation(0.01, 0, 0);
    }

    if (keyStates['d'] || keyStates['D']) {
        cameraTranslation *= Matrix::translation(-0.01, 0, 0);
    }

    if (keyStates['w'] || keyStates['W']) {
        cameraTranslation *= Matrix::translation(0, -0.01, 0);
    }

    if (keyStates['s'] || keyStates['S']) {
        cameraTranslation *= Matrix::translation(0, 0.01, 0);
    }


    if (keyStates[' '] && !lastKeyStates[' ']) {
        printf("***CLICK***\n");
        FILE *ppm = fopen("output.ppm", "wb");
        fprintf(ppm, "P6\n# This do be a file\n%d %d\n%d\n", SCREEN_WIDTH, SCREEN_HEIGHT, 255);

        for (int y = SCREEN_HEIGHT - 1; y >= 0; --y) {
            for (int x = 0; x < SCREEN_WIDTH; ++x) {
                fputc((int)(255 * pixelBuffer[(y * SCREEN_WIDTH + x) * 3]), ppm);
                fputc((int)(255 * pixelBuffer[(y * SCREEN_WIDTH + x) * 3 + 1]), ppm);
                fputc((int)(255 * pixelBuffer[(y * SCREEN_WIDTH + x) * 3 + 2]), ppm);
            }
        }

        fclose(ppm);
    }

    else {
        // glClearColor(0, 0, 0, 1);
    }


    memcpy(lastKeyStates, keyStates, 256 * sizeof(bool));
    lastMouseX = mouseX;
    lastMouseY = mouseY;


    double end = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now().time_since_epoch()).count();

    printf("%.4f\n", end - start);
}

void *rayTrace(void *args) {
    ThreadParams *params = (ThreadParams *)args;

    Vector rayDirection,
            intersection,
            l, intensity;

    for (int y = params->firstRow; y < params->firstRow + params->numRows; ++y) {
        for (int x = 0; x < SCREEN_WIDTH; ++x) {


            // before computing if there's any pixel here, clear it to the background color
            pixelBuffer[(y * SCREEN_WIDTH + x) * 3]     = 0;
            pixelBuffer[(y * SCREEN_WIDTH + x) * 3 + 1] = 0;
            pixelBuffer[(y * SCREEN_WIDTH + x) * 3 + 2] = 0;

#ifdef STIPPLE
            if ((x + y) % 2) continue;
#endif

            int     intersectionIndex    = -1;
            float   intersectionDistance = INFINITY;
            Vector *intersectionNormal;
            Vector *intersectionPoint = new Vector();

            rayDirection = (cameraDirection * rays[y][x]);

            for (int i = 0; i < numTriangles; ++i) {

                if (rayDirection.dot(*triangles[i]->normal) > 0) continue;

                float denominator = rayDirection.dot(*triangles[i]->normal);
                if (abs(denominator) < 1e-4f) continue;

                // Compute the paramtric location along the ray where the interesection with the face's plane occurs. If it is less than zero, the plane is behind us
                float t = ((*triangles[i]->a) - params->cameraLocation).dot(*triangles[i]->normal) / denominator;
                if (t <= 0 || t > FAR_PLANE_DISTANCE || t > intersectionDistance) continue;

                intersection = params->cameraLocation + rayDirection * t;

                // Keep a running total of the area of all the triangles between the face vertices and the intersection point.
                //  As soon as that becomes greater than the area of the triangle itself (plus a bit of wiggle room), continue
                float areaLeft = triangles[i]->area + FUZZ_FACTOR;
                Vector dA = intersection - *triangles[i]->a;
                Vector dB = intersection - *triangles[i]->b;
                Vector dC = intersection - *triangles[i]->c;
                // printf("%.10f\n", areaLeft);
                areaLeft -= 0.5f * (dA).cross(dB).length();
                if (areaLeft < 0) continue;
                areaLeft -= 0.5f * (dA).cross(dC).length();
                if (areaLeft < 0) continue;
                areaLeft -= 0.5f * (dB).cross(dC).length();
                if (areaLeft < 0) continue;


                intersectionDistance = t;
                intersectionIndex    = i;
                intersectionNormal   = triangles[i]->normal;
                *intersectionPoint   = intersection;
            }

            if (intersectionIndex >= 0) {
                l = (params->cameraLocation - *intersectionPoint).normalize();

#ifdef FACE_COLORS
                Vector rgb = hsvToRgb(Vector(fmod(intersectionIndex, 360), 1, 1));
                intensity  = rgb * (k_d * (l.dot(*intersectionNormal))) +
                        rgb * k_s * pow(((2 * intersectionNormal->dot(l)) * *intersectionNormal - l).dot(l), shinyness) +
                        rgb * k_a;
#else
                intensity = i_i * (k_d * (l.dot(*intersectionNormal))) +
                        i_s * k_s * pow(((2 * intersectionNormal->dot(l)) * *intersectionNormal - l).dot(l), shinyness) +
                        i_a * k_a;
#endif

                pixelBuffer[(y * SCREEN_WIDTH + x) * 3 + 0] = intensity.r;
                pixelBuffer[(y * SCREEN_WIDTH + x) * 3 + 1] = intensity.g;
                pixelBuffer[(y * SCREEN_WIDTH + x) * 3 + 2] = intensity.b;
            }

            delete intersectionPoint;
        }
    }

    return nullptr;
}