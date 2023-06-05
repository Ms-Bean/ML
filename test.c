#include "linalg/linalg.h"
struct test
{

    char a[5];
};
int main(void)
{
    Matrix m;

    double dat[4][4] = {{4,12,8,-2},{2,4,0,1},{9,18,0,1},{1,2,0,5}};
    m = m_init(4, 4);
    m_copy_c_matrix(dat, &m);

    printf("Matrix:\n");
    m_print(&m);

    m_row_echelon(&m);
    printf("\n");
    m_print(&m);
    return 0;
}