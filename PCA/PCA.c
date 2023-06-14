#include "../linalg/linalg_parallel.h"
int main(int argc, char **argv)
{
    if(argc < 2)
    {
        printf("No filename given");
        return 2;
    }
    Matrix m = m_read_csv(argv[1]);
    Matrix cov = m_create_covariance_matrix(&m);
    Matrix eigen = m_eigenvectors(&cov);

    
    for(long i = 0; i < eigen.cols; i++)
    {
        long j;
        for(j = 0; j < eigen.rows-1; j++)
        {
            printf("%lf,", m_get(&eigen, j, i));
        }
        printf("%lf\n", m_get(&eigen, j, i));
    }
    m_destroy(&m);
    m_destroy(&cov);
    m_destroy(&eigen);
    return 0;
}