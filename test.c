#include "linalg/linalg_parallel.h"
int main(void)
{
    double dat[3][3] = {{4,12,-16},{12,37,-43},{-16,-43,98}};
    Matrix m;
    Matrix L;

    m=m_init(3,3);

    m_copy_c_matrix(dat, &m);
    L = m_cholesky_factor(&m);

    m_print(&L);
    return 0;
}