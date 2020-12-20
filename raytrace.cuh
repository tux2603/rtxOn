#ifndef __CUDA_RAYTRACE_HEADER
#define __CUDA_RAYTRACE_HEADER

#include "ply.h"
#include <cuda_runtime.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

class Vector {
  public:

    CUDA_CALLABLE static Vector subtract(const Vector &a, const Vector &b);
    
    float x; /// The x component of the vector
    float y; /// The y component of the vector
    float z; /// The z component of the vector
    float w;

    float &h = x;
    float &s = y;
    float &v = z;

    float &r = x;
    float &g = y;
    float &b = z;

    float &a = w;

    /**
     * Creates a new three dimensional vector with all components set to zero
     */
    CUDA_CALLABLE Vector();

    /**
     * Creates a new three dimensional vector with the specified components
     * @param x The value for the x component
     * @param y The value for the y component
     * @param z The value for the z component
     */
    CUDA_CALLABLE Vector(float x, float y, float z);

    /**
     * Creates a new three dimensional vector with components that are 
     *  copies of the supplied vector
     * @param toCopy The vector whose components will be copied
     */
    CUDA_CALLABLE Vector(const Vector &toCopy);

    /**
     * Creates a new three dimensional vector at the given vertex
     * @param v yup
     */
    CUDA_CALLABLE Vector(const Vertex *v);

    /**
     * Delete some stuff
     */
    CUDA_CALLABLE ~Vector();

    /** Computes the cross product of this vector and the given vector b
     * @param b The vector to compute the cross product with
     * @return The cross product of this vector and b
     */
    CUDA_CALLABLE Vector cross(const Vector &b) const;

    /**
     * Computes the dot product of this vector and the given vector b
     * @param b The vector to compute the dot product with
     * @return The dot product of this vector and b
     */
    CUDA_CALLABLE float dot(const Vector &b) const;

    /** Computes the length of this vector
     * @return The length of the vector
     */
    CUDA_CALLABLE float length() const;

    /** Computes and returns a vector in the same direction as this vector, but one unit in length.
     * Does not change this Vector
     * @return A normalised version of this vector
     */
    CUDA_CALLABLE Vector normalize() const;

    CUDA_CALLABLE Vector &operator=(const Vector &toCopy);

    CUDA_CALLABLE float &operator[](size_t index);
};

CUDA_CALLABLE Vector operator+(const Vector &a, const Vector &b);
CUDA_CALLABLE Vector operator-(const Vector &a, const Vector &b);
CUDA_CALLABLE Vector operator*(const Vector &a, float scale);
CUDA_CALLABLE Vector operator*(float scale, const Vector &a);
CUDA_CALLABLE Vector operator/(const Vector &a, float denominator);

class Matrix {
  public:
    CUDA_CALLABLE static Matrix identity();

    CUDA_CALLABLE static Matrix translation(float x, float y, float z);
    CUDA_CALLABLE static Matrix scale(float x, float y, float z);

    CUDA_CALLABLE static Matrix rotateX(float theta);
    CUDA_CALLABLE static Matrix rotateY(float theta);
    CUDA_CALLABLE static Matrix rotateZ(float theta);


    float e[16];

    /**
     * Creates a new matrix object with every element set to zero
     */
    CUDA_CALLABLE Matrix();

    /**
     * Creates a new matrix object from a list of floats
     * @param elems A sixteen long array of floating point values
     */
    CUDA_CALLABLE Matrix(float *elems);

    /**
     * Creates a new matrix object from a whole lot of individual floats
     * @param e00 The matrix element at index 0,0
     * @param e01 The matrix element at index 0,1
     * @param e02 The matrix element at index 0,2
     * @param e03 The matrix element at index 0,3
     * @param e10 The matrix element at index 1,0
     * @param e11 The matrix element at index 1,1
     * @param e12 The matrix element at index 1,2
     * @param e13 The matrix element at index 1,3
     * @param e20 The matrix element at index 2,0
     * @param e21 The matrix element at index 2,1
     * @param e22 The matrix element at index 2,2
     * @param e23 The matrix element at index 2,3
     * @param e30 The matrix element at index 3,0
     * @param e31 The matrix element at index 3,1
     * @param e32 The matrix element at index 3,2
     * @param e33 The matrix element at index 3,3
     */
    CUDA_CALLABLE Matrix(float e00, float e01, float e02, float e03,
            float e10, float e11, float e12, float e13,
            float e20, float e21, float e22, float e23,
            float e30, float e31, float e32, float e33);

    CUDA_CALLABLE Matrix(const Matrix &toCopy);

    // Matrix &translate(float dx, float dy, float dz);

    CUDA_CALLABLE Matrix &operator*=(const Matrix &b);
    CUDA_CALLABLE Matrix  operator*(const Matrix &b) const;
};

CUDA_CALLABLE Vector operator*(Matrix a, Vector b);

typedef struct Triangle {
    Vector a, b, c;
    Vector *normal;
    float   area;
    Triangle() : a(Vector()), b(Vector()), c(Vector()), normal(new Vector()), area(0) {};
    Triangle(Vector a,
            Vector b,
            Vector c) : a(Vector(a)),
                         b(Vector(b)),
                         c(Vector(c)),
                         normal(new Vector((a - b).cross(b - c).normalize())),
                         area(0.5f * (a - b).cross((a - c)).length()){};
} Triangle;

typedef struct LightData {
    Vector diffuseColor;
    Vector specularColor;
    Vector ambientColor;
    float  diffuseIntensity;
    float  specularIntensity;
    float  ambientIntensity;
    float  shininess;

    LightData(Vector diffuseColor,
            Vector   specularColor,
            Vector   ambientColor,
            float    diffuseIntensity,
            float    specularIntensity,
            float    ambientIntensity,
            float    shininess) : diffuseColor(diffuseColor),
                               specularColor(specularColor),
                               ambientColor(ambientColor),
                               diffuseIntensity(diffuseIntensity),
                               specularIntensity(specularIntensity),
                               ambientIntensity(ambientIntensity),
                               shininess(shininess){};

} LightData;

typedef struct CameraData {
    Vector location;
    Matrix direction;
    float  imagePlaneDistance;

    CameraData(Vector location,
            Matrix    direction,
            float     imagePlaneDistance) : location(location),
                                        direction(direction),
                                        imagePlaneDistance(imagePlaneDistance){};
} CameraData;

void drawCircleKernel(float *pixels, int width, int height, int blockWidth, int blockHeight);
void drawRaytraceKernel(float *pixels, Triangle *triangles, int numTriangles, int width, int height, CameraData camera, LightData light);
#endif