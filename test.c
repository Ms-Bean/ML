#include "linalg/linalg_parallel.h"
int main(void)
{
    double dat[3][3] = {{-2,-4,2},{-2,1,2},{4,2,5}};

    Matrix m = m_init(3, 3);
    m_copy_c_matrix(dat, &m);
    Matrix e = m_eigenvalues(&m, 100);

    m_label_print(&m, "Matrix");
    m_label_print(&e, "Eigenvalues");
    return 0;
}