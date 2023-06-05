#include "linalg/linalg.h"
struct test
{

    char a[5];
};
int main(void)
{
    Matrix m;

    double dat[4][4] = {{1,-2,1,-1},{-4,-2,0,4},{2,3,-1,-3},{17,-10,11,1}};
    m = m_init(4, 4);
    m_copy_c_matrix(dat, &m);

    printf("Matrix:\n");
    m_print(&m);
    printf("\nLinearly independent columns: %s\n", m_row_linear_independent(&m) ? "Yes" : "No");
    printf("Rank: %ld\n", m_rank(&m));
    return 0;
}