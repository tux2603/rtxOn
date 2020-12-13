#include "utils.h"
#include <cstring>
#include <math.h>


// #define DEBUG

Vector::Vector() : Vector(0, 0, 0) {}
Vector::Vector(float x, float y, float z) : x(x), y(y), z(z), w(1) {}
Vector::Vector(const Vector &toCopy) : Vector(toCopy.x, toCopy.y, toCopy.z) {}
Vector::Vector(const Vertex *v) : Vector(v->x, v->y, v->z) {}

Vector::~Vector() {
#ifdef DEBUG
    std::cout << "Deleted vector " << *this << std::endl;
#endif
}

Vector &Vector::operator=(const Vector &toCopy) {
    x = toCopy.x;
    y = toCopy.y;
    z = toCopy.z;
    w = 1;

    return *this;
}

float &Vector::operator[](size_t index) {
    switch (index) {
        case 0:
            return x;
            break;
        case 1:
            return y;
            break;
        case 2:
            return z;
            break;
        case 3:
            return w;
            break;
    }
}

Vector Vector::cross(const Vector &b) const {
    float newX = y * b.z - z * b.y;
    float newY = z * b.x - x * b.z;
    float newZ = x * b.y - y * b.x;
    return Vector(newX, newY, newZ);
}

float Vector::dot(const Vector &b) const {
    return x * b.x + y * b.y + z * b.z;
}

float Vector::length() const {
    return sqrt(x * x + y * y + z * z);
}

Vector Vector::normalize() const {
    return *this / length();
}

Vector operator+(const Vector &a, const Vector &b) {
    return Vector(a.x + b.x, a.y + b.y, a.z + b.z);
}

Vector operator-(const Vector &a, const Vector &b) {
    return Vector(a.x - b.x, a.y - b.y, a.z - b.z);
}

Vector operator*(const Vector &a, float scale) {
    return Vector(a.x * scale, a.y * scale, a.z * scale);
}

Vector operator*(float scale, const Vector &a) {
    return Vector(a.x * scale, a.y * scale, a.z * scale);
}

Vector operator/(const Vector &a, float denominator) {
    return Vector(a.x / denominator, a.y / denominator, a.z / denominator);
}

std::ostream &operator<<(std::ostream &os, const Vector &v) {
    return os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
}



// #######################
// ##### MATRIX CODE #####
// #######################


Matrix Matrix::identity() {
    return Matrix(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
}

Matrix Matrix::translation(float x, float y, float z) {
    return Matrix(1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z, 0, 0, 0, 1);
}

Matrix Matrix::scale(float x, float y, float z) {
    return Matrix(x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1);
}

Matrix Matrix::rotateX(float theta) {
    float c = std::cos(theta),
          s = std::sin(theta);
    return Matrix(1, 0, 0, 0, 0, c, -s, 0, 0, s, c, 0, 0, 0, 0, 1);
}

Matrix Matrix::rotateY(float theta) {
    float c = std::cos(theta),
          s = std::sin(theta);
    return Matrix(c, 0, s, 0, 0, 1, 0, 0, -s, 0, c, 0, 0, 0, 0, 1);
}

Matrix Matrix::rotateZ(float theta) {
    float c = std::cos(theta),
          s = std::sin(theta);
    return Matrix(c, -s, 0, 0, s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
}

Matrix::Matrix() {
    for (int i = 0; i < 16; ++i)
        e[i] = 0;
}

Matrix::Matrix(float *elems) {
    memcpy(e, elems, 16 * sizeof(float));
}

// Hopefully this monstrosity will never be used
Matrix::Matrix(float e00, float e01, float e02, float e03,
        float e10, float e11, float e12, float e13,
        float e20, float e21, float e22, float e23,
        float e30, float e31, float e32, float e33) {
    e[0]  = e00;
    e[1]  = e01;
    e[2]  = e02;
    e[3]  = e03;
    e[4]  = e10;
    e[5]  = e11;
    e[6]  = e12;
    e[7]  = e13;
    e[8]  = e20;
    e[9]  = e21;
    e[10] = e22;
    e[11] = e23;
    e[12] = e30;
    e[13] = e31;
    e[14] = e32;
    e[15] = e33;
}

Matrix::Matrix(const Matrix &toCopy) {
    memcpy(e, toCopy.e, 16 * sizeof(float));
}

Matrix &Matrix::operator*=(const Matrix &b) {
    float *newElems = new float[16];

    for (int x = 0; x < 4; ++x) {
        for (int y = 0; y < 4; ++y) {

            newElems[y * 4 + x] = 0;

            for (int i = 0; i < 4; ++i) {
                newElems[y * 4 + x] += e[y * 4 + i] * b.e[i * 4 + x];
            }
        }
    }

    memcpy(e, newElems, 16 * sizeof(float));

    delete[] newElems;

    return *this;
}

Matrix Matrix::operator*(const Matrix &b) const {
    float *newElems = new float[16];

    for (int x = 0; x < 4; ++x) {
        for (int y = 0; y < 4; ++y) {

            newElems[y * 4 + x] = 0;

            for (int i = 0; i < 4; ++i) {
                newElems[y * 4 + x] += e[y * 4 + i] * b.e[i * 4 + x];
            }
        }
    }


    Matrix newMatrix = Matrix(newElems);
    delete[] newElems;

    return newMatrix;
}

Vector operator*(const Matrix a, Vector b) {
    float x = 0, y = 0, z = 0, w = 0;

    for (int i = 0; i < 4; ++i) {
        x += a.e[i] * b[i];
        y += a.e[4 + i] * b[i];
        z += a.e[8 + i] * b[i];
        w += a.e[12 + i] * b[i];
    }

    return Vector(x / w, y / w, z / w);
}

// Equation found on wikipedia
Vector hsvToRgb(Vector hsv) {
    float c = hsv.s * hsv.v;
    float x = c * (1.f - abs(fmod(hsv.h / 60.f, 2) - 1.f));
    float m = hsv.v - c;

    Vector rgb;

    int sextant = (int)hsv.h / 60;

    switch (sextant) {
        case 0:
            rgb = Vector(c, x, 0);
            break;
        case 1:
            rgb = Vector(x, c, 0);
            break;
        case 2:
            rgb = Vector(0, c, x);
            break;
        case 3:
            rgb = Vector(0, x, c);
            break;
        case 4:
            rgb = Vector(x, 0, c);
            break;
        case 5:
            rgb = Vector(c, 0, x);
            break;
    }

    return (rgb + Vector(m, m, m));
}
