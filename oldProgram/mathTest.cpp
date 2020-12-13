#include <iostream>

#include "utils.h"

int main() {
    Vector a(1, 2, 3);
    Vector b;
    b = a * 2;
    Vector c = 3 * a;

    std::cout << a << " " << b << " " << c << std::endl;

    std::cout << a + b << std::endl;
}