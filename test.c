#include "linalg/linalg_parallel.h"
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/time.h>

int main(void)
{
    Matrix m1;
    Matrix m2;
    Matrix mult;

    m1 = m_init(1000, 1000);
    m2 = m_init(1000, 1000);

    for(long i = 0; i < 1000000; i++)
    {
        m1.contents[i] = i / 10000;
        m2.contents[i] = (1000000 - i) / 10000;
    }

    mult = m_init(1000, 1000);

    long A[400][400];
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    m_mult(&m1, &m2, &mult);
    gettimeofday(&stop, NULL);
    printf("Time spent on regular mult: %f\n", (stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) / 1000000);
    
    gettimeofday(&start, NULL);
    m_mult_parallel(&m2, &m1, &mult);
    gettimeofday(&stop, NULL);
    printf("Time spent on parallel mult: %f\n", (stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) / 1000000);

    return 0;
}