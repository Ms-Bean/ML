#include "linalg/linalg.h"
#include "<time.h>"
Matrix m_PCA_dimensionality_reduction(Matrix *src, Matrix *proj, long k);



/*
Takes in a standardized data matrix, and reduces to k dimensions. 
proj should be pointer to uninitialized matrix
Compute v@proj to project column vector v onto the lower-dimensional plane
*/
Matrix m_PCA_dimensionality_reduction(Matrix *src, Matrix *proj, long k)
{
    Matrix covariance_matrix = m_create_covariance_matrix(src);
    Matrix eigenvector_matrix = m_eigenvectors(&covariance_matrix);
    m_destroy(&covariance_matrix);
    *proj = m_init(eigenvector_matrix.rows, k);

    for(long i = 0; i < proj->rows; i++)
    {
        for(long j = 0; j < k; j++)
        {
            proj->contents[i * proj->cols + j] = eigenvector_matrix.contents[i * eigenvector_matrix.cols + j];
        }
    }
    m_destroy(&eigenvector_matrix);

    return m_mult(src, proj);
}
