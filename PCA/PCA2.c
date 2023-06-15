#include "../linalg/linalg_parallel.h"
int main(int argc, char **argv)
{
    if(argc < 2)
    {
        printf("No filename given");
        return 2;
    }
    Matrix m = m_read_csv(argv[1]);
    m_standardize(&m);
    
    Matrix Ut;
    Matrix reduced = m_PCA_dimensionality_reduction(&m, &Ut, 2);
    printf("a,b\n");
    for(long i = 0; i < reduced.rows; i++)
    {
        long j;
        for(j = 0; j < reduced.cols-1; j++)
        {
            printf("%lf,", m_get(&reduced, i, j));
        }
        printf("%lf\n", m_get(&reduced, i, j));
    }

    return 0;
}