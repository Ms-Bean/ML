#include <stdio.h>
#include <stdlib.h>
#include "linalg/mat.h"
int main(void)
{
    Matrix vector;
    Matrix identity;
    Matrix product;

    double a[5] = {1,2,3,4,5};

    vector = m_init(1, 5);
    m_copy_c_matrix(a, &vector);
    identity = m_create_identity_matrix(5);

    product = m_init(1, 5);
    m_mult(&vector, &identity, &product);
    m_print(&product);
    return 0;
}