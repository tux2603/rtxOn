#include "raytrace.cuh"

// #include <__clang_cuda_builtin_vars.h>
#include <bits/stdint-uintn.h>
#include <math.h>

#define BLOCK_SIZE 16
#define PIXELS_PER_BLOCK 256
#define TRIANGLES_PER_ITERATION 256

#define FAR_PLANE_DISTANCE 1e3
#define FUZZ_FACTOR 1e-5f
#define MINIMUM_DENOMINATOR 1e-4f

#define MAP(x, srcLow, srcHigh, destLow, destHigh) (((x) - (srcLow)) * ((destHigh) - (destLow)) / ((srcHigh) - (srcLow)) + (destLow))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define DOT(a, b) ((a).x * (b).x + (a).y * (b).y + (a).z * (b).z)

__host__ __device__ Vector::Vector() : x(0), y(0), z(0), w(1) {}
__host__ __device__ Vector::Vector(float x, float y, float z) : x(x), y(y), z(z), w(1) {}
__host__ __device__ Vector::Vector(const Vector &toCopy) : x(toCopy.x), y(toCopy.y), z(toCopy.z) {}
__host__ __device__ Vector::Vector(const Vertex *v) : x(v->x), y(v->y), z(v->z) {}

__host__ __device__ Vector::~Vector() {}

__host__ __device__ Vector &Vector::operator=(const Vector &toCopy) {
    x = toCopy.x;
    y = toCopy.y;
    z = toCopy.z;
    w = 1;

    return *this;
}

__host__ __device__ float &Vector::operator[](size_t index) {
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
        default:
            return w;
            break;
    }
}

__host__ __device__ Vector Vector::cross(const Vector &b) const {
    float newX = y * b.z - z * b.y;
    float newY = z * b.x - x * b.z;
    float newZ = x * b.y - y * b.x;
    return Vector(newX, newY, newZ);
}

__host__ __device__ float Vector::dot(const Vector &b) const {
    return x * b.x + y * b.y + z * b.z;
}

__host__ __device__ float Vector::length() const {
    return sqrt(x * x + y * y + z * z);
}

__host__ __device__ Vector Vector::normalize() const {
    return *this / length();
}

__host__ __device__ Vector operator+(const Vector &a, const Vector &b) {
    return Vector(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ Vector Vector::subtract(const Vector &a, const Vector &b) {
    return Vector(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ Vector operator-(const Vector &a, const Vector &b) {
    return Vector::subtract(a, b);
}

__host__ __device__ Vector operator*(const Vector &a, float scale) {
    return Vector(a.x * scale, a.y * scale, a.z * scale);
}

__host__ __device__ Vector operator*(float scale, const Vector &a) {
    return Vector(a.x * scale, a.y * scale, a.z * scale);
}

__host__ __device__ Vector operator/(const Vector &a, float denominator) {
    return Vector(a.x / denominator, a.y / denominator, a.z / denominator);
}

__host__ __device__ Matrix Matrix::identity() {
    return Matrix(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
}

__host__ __device__ Matrix Matrix::translation(float x, float y, float z) {
    return Matrix(1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z, 0, 0, 0, 1);
}

__host__ __device__ Matrix Matrix::scale(float x, float y, float z) {
    return Matrix(x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1);
}

__host__ __device__ Matrix Matrix::rotateX(float theta) {
    float c = cos(theta),
          s = sin(theta);
    return Matrix(1, 0, 0, 0, 0, c, -s, 0, 0, s, c, 0, 0, 0, 0, 1);
}

__host__ __device__ Matrix Matrix::rotateY(float theta) {
    float c = cos(theta),
          s = sin(theta);
    return Matrix(c, 0, s, 0, 0, 1, 0, 0, -s, 0, c, 0, 0, 0, 0, 1);
}

__host__ __device__ Matrix Matrix::rotateZ(float theta) {
    float c = cos(theta),
          s = sin(theta);
    return Matrix(c, -s, 0, 0, s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
}

__host__ __device__ Matrix::Matrix() {
    for (int i = 0; i < 16; ++i)
        e[i] = 0;
}

__host__ __device__ Matrix::Matrix(float *elems) {
    memcpy(e, elems, 16 * sizeof(float));
}

// Hopefully this monstrosity will never be used
__host__ __device__ Matrix::Matrix(float e00, float e01, float e02, float e03,
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

__host__ __device__ Matrix::Matrix(const Matrix &toCopy) {
    memcpy(e, toCopy.e, 16 * sizeof(float));
}

__host__ __device__ Matrix &Matrix::operator*=(const Matrix &b) {
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

__host__ __device__ Matrix Matrix::operator*(const Matrix &b) const {
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

__host__ __device__ Vector operator*(const Matrix a, Vector b) {
    float x = 0, y = 0, z = 0, w = 0;

    for (int i = 0; i < 4; ++i) {
        x += a.e[i] * b[i];
        y += a.e[4 + i] * b[i];
        z += a.e[8 + i] * b[i];
        w += a.e[12 + i] * b[i];
    }

    return Vector(x / w, y / w, z / w);
}


typedef struct sVector {
    float x, y, z;

    __device__ sVector() : x(0), y(0), z(0){};
    __device__ sVector(float x, float y, float z) : x(x), y(y), z(z){};
    __device__ sVector(Vector a) : x(a.x), y(a.y), z(a.z){};
} sVector;

__device__ float length(sVector a) {
    return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

__device__ sVector cross(sVector a, sVector b) {
    float x = a.y * b.z - a.z * b.y;
    float y = a.z * b.x - a.x * b.z;
    float z = a.x * b.y - a.y * b.x;
    return sVector(x, y, z);
}

__device__ sVector crossNormal(sVector a, sVector b) {
    float x = a.y * b.z - a.z * b.y;
    float y = a.z * b.x - a.x * b.z;
    float z = a.x * b.y - a.y * b.x;
    float s = 1 / sqrt(x * x + y * y + z * z);
    return sVector(x * s, y * s, z * s);
}

__device__ float crossLength(sVector a, sVector b) {
    float x = a.y * b.z - a.z * b.y;
    float y = a.z * b.x - a.x * b.z;
    float z = a.x * b.y - a.y * b.x;
    return sqrt(x * x + y * y + z * z);
}

__device__ sVector normalize(sVector a) {
    float length = sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    a.x /= length;
    a.y /= length;
    a.z /= length;
    return a;
}

__device__ sVector scale(sVector a, float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    return a;
}
__device__
        sVector
        subtract(sVector a, sVector b) {
    return sVector(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ sVector transform(Matrix m, sVector a) {
    float x = m.e[0] * a.x + m.e[1] * a.y + m.e[2] * a.z + m.e[3],
          y = m.e[4] * a.x + m.e[5] * a.y + m.e[6] * a.z + m.e[7],
          z = m.e[8] * a.x + m.e[9] * a.y + m.e[10] * a.z + m.e[11],
          w = m.e[12] * a.x + m.e[13] * a.y + m.e[14] * a.z + m.e[15];

    sVector v = sVector(x / w, y / w, z / w);
    return v;
}

typedef struct sTriangle {
    sVector a, b, c;
    sVector normal;

    __device__ sTriangle() : a(sVector()), b(sVector()), c(sVector()), normal(sVector(1, 0, 0)){};
    __device__ sTriangle(sVector a, sVector b, sVector c) : a(a), b(b), c(c), normal(crossNormal(subtract(a, b), subtract(a, c))){};
} sTriangle;


// ############################################################################
// #####                        END OF MATHY STUFF                        #####
// ############################################################################



__global__ void circleKernel(float *pixelData, int width, int height) {
    float denominator = 1 / (float)MIN(width, height);
    float halfWidth   = width / 2.f;
    float halfHeight  = height / 2.f;

    for (int x = threadIdx.x; x < width; x += blockDim.x) {
        for (int y = threadIdx.y; y < height; y += blockDim.y) {
            float dx                = x - halfWidth;
            float dy                = y - halfHeight;
            float intensity         = sqrt(dx * dx + dy * dy) * denominator;
            int   pixelIdx          = 3 * (x + y * width);
            pixelData[pixelIdx]     = intensity;
            pixelData[pixelIdx + 1] = intensity;
            pixelData[pixelIdx + 2] = intensity;
        }
    }
}


__global__ void raytraceKernel(float *pixelData, Triangle *triangles, int numTriangles, int width, int height, CameraData camera, LightData light) {
    __shared__ int     indexes[PIXELS_PER_BLOCK];
    __shared__ float   distances[PIXELS_PER_BLOCK];
    __shared__ sVector normals[PIXELS_PER_BLOCK];
    __shared__ float   pixels[PIXELS_PER_BLOCK * 3];

    // printf("Here I am on line %d\n", __LINE__);

    uint pixelIdx       = threadIdx.x + threadIdx.y * blockDim.x;
    uint globalPixelIdx = threadIdx.x + blockIdx.x * blockDim.x + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;


    // printf("Here I am on line %d\n", __LINE__);

    sVector ray = sVector(MAP(threadIdx.x + blockIdx.x * blockDim.x, 0, width - 1, -1.f, 1.f), MAP(threadIdx.y + blockIdx.y * blockDim.y, 0, width - 1, -1.f, 1.f), -camera.imagePlaneDistance);
    ray         = normalize(transform(camera.direction, ray));


    // printf("Here I am on line %d\n", __LINE__);

    pixels[pixelIdx * 3 + 0] = 0;
    pixels[pixelIdx * 3 + 1] = 0;
    pixels[pixelIdx * 3 + 2] = 0;

    sVector cameraLocation = sVector(camera.location.x, camera.location.y, camera.location.z);


    distances[pixelIdx] = FAR_PLANE_DISTANCE + 1;
    indexes[pixelIdx]   = -1;


    // printf("Here I am on line %d\n", __LINE__);

    for (int i = 0; i < numTriangles; i += blockDim.z) {



        // printf("Here I am on line %d\n", __LINE__);
        // // Copy in this bunch of triangles
        // if (pixelIdx + threadIdx.z * PIXELS_PER_BLOCK < TRIANGLES_PER_ITERATION) {
        //     Triangle *triangle = triangles[globalIdx + pixelIdx + threadIdx.z * PIXELS_PER_BLOCK];

        //     triangleBuffer[pixelIdx + threadIdx.z * PIXELS_PER_BLOCK].a = sVector(triangle->a->x, triangle->a->y, triangle->a->z);
        //     triangleBuffer[pixelIdx + threadIdx.z * PIXELS_PER_BLOCK].b = sVector(triangle->b->x, triangle->b->y, triangle->b->z);
        //     triangleBuffer[pixelIdx + threadIdx.z * PIXELS_PER_BLOCK].c = sVector(triangle->c->x, triangle->c->y, triangle->c->z);
        // }

        // __syncthreads();

        // // Check all of the triangles for intersections
        // for (int i = 0; i < TRIANGLES_PER_ITERATION; i += blockDim.z) {
        //     sVector normal = triangleBuffer[i].normal;
        //     // float denominator = DOT(ray, normal);

        //     // // TODO: is there a non
        //     // subtract(normal, cameraLocation);

        //     __syncthreads();
        // }

        sTriangle t = sTriangle(sVector(triangles[i].a.x, triangles[i].a.y, triangles[i].a.z),
                sVector(triangles[i].b.x, triangles[i].b.y, triangles[i].b.z),
                sVector(triangles[i].c.x, triangles[i].c.y, triangles[i].c.z));

        // sTriangle t = sTriangle(sVector(rawTriangle.a.x, 0, 5), sVector(4, 0, 0), sVector(0, 5, 0));


        // printf("Here I am on line %d\n", __LINE__);

        float   denominator = DOT(ray, t.normal);
        sVector temp        = subtract(t.b, cameraLocation);


        // printf("Triangle (%f, %f, %f)  (%f, %f, %f)  (%f, %f, %f)\n",
        //         triangles[i].a.x, triangles[i].a.y, triangles[i].a.z,
        //         triangles[i].b.x, triangles[i].b.y, triangles[i].b.z,
        //         triangles[i].c.x, triangles[i].c.y, triangles[i].c.z);
        // printf("%d: (%f, %f, %f) dot (%f, %f, %f)\n", i, temp.x, temp.y, temp.z, t.normal.x, t.normal.y, t.normal.z);
        float distance = DOT(subtract(t.a, cameraLocation), t.normal) / denominator;


        // printf("Here I am on line %d\n", __LINE__);

        sVector intersection = sVector(cameraLocation.x + ray.x * distance, cameraLocation.y + ray.y * distance, cameraLocation.z + ray.z * distance);

        sVector deltaA = subtract(intersection, t.a);
        sVector deltaB = subtract(intersection, t.b);
        sVector deltaC = subtract(intersection, t.c);

        float area    = 0.5f * (crossLength(deltaA, deltaB) + crossLength(deltaB, deltaC) + crossLength(deltaA, deltaC));
        float maxArea = 0.5f * crossLength(subtract(t.a, t.b), subtract(t.a, t.c));

        // printf("Here I am on line %d\n", __LINE__);

        // printf("%d: distance = %f, denom = %f, %f vs %f\n", i, distance * 1000, denominator, area, maxArea);

        // for (int j = 0; j < blockDim.z; ++j) {
        //     if (threadIdx.z == j) {


        if (denominator > 0 && distance >= 0 && distance <= FAR_PLANE_DISTANCE && distance < distances[pixelIdx] && area < maxArea + FUZZ_FACTOR) {
            distances[pixelIdx] = distance;
            normals[pixelIdx]   = t.normal;
            indexes[pixelIdx]   = i;
        }
        //     }

        __syncthreads();
        // }
    }

    if (indexes[pixelIdx] >= 0) {
        // sVector l = normalize(subtract(cameraLocation, intersections[pixelIdx]));

        //pow(DOT(subtract(cross(scale(ray, 2), normals[pixelIdx]), ray), ray), light.shininess)

        sVector diffuse = scale(sVector(light.diffuseColor), DOT(ray, normals[pixelIdx]) * light.diffuseIntensity);
        // ((2 * intersectionNormal->dot(l)) * *intersectionNormal
        sVector specular = scale(sVector(light.specularColor), pow(DOT(subtract(scale(normals[pixelIdx], 2 * DOT(normals[pixelIdx], ray)), ray), ray), light.shininess) * light.specularIntensity);
        sVector ambient  = scale(light.ambientColor, light.ambientIntensity);


        pixels[pixelIdx * 3 + 0] = diffuse.x + specular.x + ambient.x;
        pixels[pixelIdx * 3 + 1] = diffuse.y + specular.y + ambient.y;
        pixels[pixelIdx * 3 + 2] = diffuse.z + specular.z + ambient.z;
    }

    pixelData[globalPixelIdx * 3 + 0] = pixels[pixelIdx * 3 + 0];
    pixelData[globalPixelIdx * 3 + 1] = pixels[pixelIdx * 3 + 1];
    pixelData[globalPixelIdx * 3 + 2] = pixels[pixelIdx * 3 + 2];
}

void drawCircleKernel(float *pixels, int width, int height, int blockWidth, int blockHeight) {
    circleKernel<<<dim3(width / blockWidth, height / blockHeight, 1), dim3(blockWidth, blockHeight, 1)>>>(pixels, width, height);
    // cudaDeviceSynchronize();
}

void drawRaytraceKernel(float *pixels, Triangle *triangles, int numTriangles, int width, int height, CameraData camera, LightData light) {
    uint blockSizeSquared = BLOCK_SIZE * BLOCK_SIZE;
    uint sharedMemSize    = sizeof(int) * blockSizeSquared + sizeof(float) * blockSizeSquared + sizeof(sVector) * blockSizeSquared * 2;
    printf("Creating new kernel, %u bytes of shared memory per block\n", sharedMemSize);

    raytraceKernel<<<dim3(width / BLOCK_SIZE, width / BLOCK_SIZE, 1), dim3(BLOCK_SIZE, BLOCK_SIZE, 1)>>>(pixels, triangles, numTriangles, width, height, camera, light);
    // raytraceKernel<<<dim3(width / BLOCK_SIZE, width / BLOCK_SIZE, 1), dim3(BLOCK_SIZE, BLOCK_SIZE, 1)>>>(pixels, triangles, 5, width, height, camera, light);
    // raytraceKernel<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(pixels, triangles, numTriangles, width, height, camera, light);

    cudaDeviceSynchronize();
    printf("Kernel call completed\n");

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}
