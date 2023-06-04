#include <stdio.h>
#include <stdlib.h>
#include "linalg/linalg.h"
int main(void)
{
    Matrix m;
    Matrix solution;
    double dat[3][4] = {{2,1,1,7},{2,-1,2,6},{1,-2,1,0}};

    m = m_init(3, 4);
    m_copy_c_matrix(dat, &m);
    printf("Augmented matrix:\n");
    m_print(&m);
    printf("\nReduced row-echelon form:\n");
    m_reduced_echelon(&m);
    m_print(&m);
    
    printf("\nSolution:\n"); 
    solution = m_back_substitution(&m);
    m_print(&solution);
    return 0;
}