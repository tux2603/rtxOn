#include "ply.h"
#include <iostream>

#ifndef _UTILS_INCLUDE_GUARD
#define _UTILS_INCLUDE_GUARD

/**
 * A class that implements a simple three dimensional vector and some basic mathematical operations.
 * Nothing fancy really
 */
class Vector {
  public:
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
    Vector();

    /**
     * Creates a new three dimensional vector with the specified components
     * @param x The value for the x component
     * @param y The value for the y component
     * @param z The value for the z component
     */
    Vector(float x, float y, float z);

    /**
     * Creates a new three dimensional vector with components that are 
     *  copies of the supplied vector
     * @param toCopy The vector whose components will be copied
     */
    Vector(const Vector &toCopy);

    /**
     * Creates a new three dimensional vector at the given vertex
     * @param v yup
     */
    Vector(const Vertex *v);

    ~Vector();

    /** Computes the cross product of this vector and the given vector b
     * @param b The vector to compute the cross product with
     * @return The cross product of this vector and b
     */
    Vector cross(const Vector &b) const;

    /**
     * Computes the dot product of this vector and the given vector b
     * @param b The vector to compute the dot product with
     * @return The dot product of this vector and b
     */
    float dot(const Vector &b) const;

    /** Computes the length of this vector
     * @return The length of the vector
     */
    float length() const;

    /** Computes and returns a vector in the same direction as this vector, but one unit in length.
     * Does not change this Vector
     * @return A normalised version of this vector
     */
    Vector normalize() const;

    Vector &operator=(const Vector &toCopy);

    float &operator[](size_t index);
};

/**
 * Prints the vector out to the given stream in a nice human-readable format
 * @param os The stream to print the vector to
 * @param v The vector to print to the stream
 * @return The stream after printing the vector
 */
std::ostream &operator<<(std::ostream &os, const Vector &v);

Vector operator+(const Vector &a, const Vector &b);
Vector operator-(const Vector &a, const Vector &b);
Vector operator*(const Vector &a, float scale);
Vector operator*(float scale, const Vector &a);
Vector operator/(const Vector &a, float denominator);

class Matrix {
  public:
    static Matrix identity();

    static Matrix translation(float x, float y, float z);
    static Matrix scale(float x, float y, float z);
    static Matrix rotateX(float theta);
    static Matrix rotateY(float theta);
    static Matrix rotateZ(float theta);


    float e[16];

    /**
     * Creates a new matrix object with every element set to zero
     */
    Matrix();

    /**
     * Creates a new matrix object from a list of floats
     * @param elems A sixteen long array of floating point values
     */
    Matrix(float *elems);

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
    Matrix(float e00, float e01, float e02, float e03,
            float e10, float e11, float e12, float e13,
            float e20, float e21, float e22, float e23,
            float e30, float e31, float e32, float e33);

    Matrix(const Matrix &toCopy);

    // Matrix &translate(float dx, float dy, float dz);

    Matrix &operator*=(const Matrix &b);
    Matrix  operator*(const Matrix &b) const;

  private:
};


Vector operator*(Matrix a, Vector b);

typedef struct Triangle {
    Vector *a, *b, *c;
    Vector *normal;
    float   area;
    Triangle(Vector *a, Vector *b, Vector *c) : a(a), b(b), c(c),
                                                normal(new Vector((*a - *b).cross(*b - *c).normalize())),
                                                area(0.5f * (*a - *b).cross((*a - *c)).length()){};
} Triangle;

Vector hsvToRgb(Vector hsv);

#endif