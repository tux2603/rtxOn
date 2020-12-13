#include <GL/gl.h>
#include <GL/glut.h>

#include <chrono>
#include <string.h>

#include "ply.h"

#define SCREEN_WIDTH 256
#define SCREEN_HEIGHT 256

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/// Sets up the window and all necessary buffers et all
void init(char *modelName);

/// Draws the scene as necessary
void draw();

/// Draws a string at the specified location
void drawString(float x, float y, char *str, bool large = true);

/// Shared memory between host and CPU
float *pixelBuffer;

/// The number of faces in the model
int numFaces = 0;

/// The number of vertices in the model
int numVertices = 0;

/// An array of all the raw vertex data in the model
Vertex **vlist = nullptr;

/// An array of all the raw face data in the model
Face **flist = nullptr;



// ############################################################################
// #####                        END OF DEFINITIONS                        #####
// ############################################################################



int main(int argc, char **argv) {
    char *modelName = (argc > 1 ? argv[1] : (char *)"ply/bunny.ply");

    glutInit(&argc, argv);
    init(modelName);
}

void init(char *modelName) {

    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);

    glutInitWindowPosition(50, 100);
    glutInitWindowSize(SCREEN_WIDTH, SCREEN_HEIGHT);

    // Set the title of the window to show the file being displayed
    char *titleBuffer = (char *)myalloc(sizeof(char) * (strlen("CUDA ray tracing | ") + strlen(modelName) + 1));
    sprintf(titleBuffer, "CUDA ray tracing | %s", modelName);
    glutCreateWindow(titleBuffer);

    
    glMatrixMode(GL_PROJECTION);
    glOrtho(0, SCREEN_WIDTH, 0, SCREEN_HEIGHT, -1, 1);

    glClearColor(0, 0, 0, 1);

    pixelBuffer = new GLfloat[SCREEN_WIDTH * SCREEN_HEIGHT * 3];

    for(int i = 0; i < SCREEN_HEIGHT * SCREEN_WIDTH * 3; ++i) {
        pixelBuffer[i] = i / (float)(SCREEN_HEIGHT * SCREEN_WIDTH * 3);
        pixelBuffer[i + 1] = i / (float)(SCREEN_HEIGHT * SCREEN_WIDTH * 3);
        pixelBuffer[i + 2] = i / (float)(SCREEN_HEIGHT * SCREEN_WIDTH * 3);
    }

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

            minX = MIN(minX, vert->x);
            minY = MIN(minY, vert->y);
            minZ = MIN(minZ, vert->z);


            maxX = MAX(maxX, vert->x);
            maxY = MAX(maxY, vert->y);
            maxZ = MAX(maxZ, vert->z);
        }
    }

    float deltaX = maxX - minX;
    float deltaY = maxY - minY;
    float deltaZ = maxZ - minZ;

    float maxDelta = MAX(deltaX, MAX(deltaY, deltaZ));

    glutDisplayFunc(draw);
    glutIdleFunc(glutPostRedisplay);

    glutMainLoop();

    free(titleBuffer);
}

void draw() {
    // Just add in some quick and dirty FPS calculations...
    static double lastTime   = 0;
    static int    numFrames  = 0;
    static int    currentFPS = 1;

    double timeNow = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now().time_since_epoch()).count();
    ++numFrames;

    if (timeNow - lastTime > 1) {
        lastTime   = timeNow;
        currentFPS = numFrames;
        numFrames  = 0;
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glRasterPos2f(0, 0);
    glDrawPixels(SCREEN_WIDTH, SCREEN_HEIGHT, GL_RGB, GL_FLOAT, (GLvoid *)pixelBuffer);

    char *fpsString = (char *)myalloc(sizeof(char) * (8 + strlen(" FPS")));
    sprintf(fpsString, "%d FPS", currentFPS);
    drawString(4, 4, fpsString, false);
    free(fpsString);

    glFlush();
    glutSwapBuffers();
}

void drawString(float x, float y, char *str, bool large) {
    glColor3f(1, 1, 1);

    glRasterPos2f(x, y);
    for (size_t i = 0; i < strlen(str); ++i)
        glutBitmapCharacter(large ? GLUT_BITMAP_9_BY_15 : GLUT_BITMAP_HELVETICA_10, str[i]);
    glEnd();
}
