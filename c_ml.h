#include "linalg/linalg.h"
#include <time.h>
#include <limits.h>

Matrix PCA_dimensionality_reduction(Matrix *src, Matrix *proj, long k);
Matrix k_means_clustering(Matrix *src, long k);
long k_means_predict(Matrix *point, Matrix *centroid_matrix);


/*
Takes in a standardized data matrix, and reduces to k dimensions. 
proj should be pointer to uninitialized matrix
Compute v@proj to project column vector v onto the lower-dimensional plane
*/
Matrix PCA_dimensionality_reduction(Matrix *src, Matrix *proj, long k)
{
    Matrix covariance_matrix = m_create_covariance_matrix(src);
    Matrix eigenvector_matrix = m_eigenvectors(&covariance_matrix, 1000);
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
/*Takes in a data matrix and k, returns data matrix containing the centroids of each cluster*/
Matrix k_means_clustering(Matrix *src, long k)
{
    if(src->rows < k)
    {
        printf("clusters greater than number of data points");
        exit(3);
    }
    srand(time(NULL));

    Matrix old_centroids = m_init(k, src->cols);
    Matrix new_centroids = m_init(k, src->cols);

    long *classifications = calloc(src->rows, sizeof(long));
    long *starting_indices = malloc(sizeof(long) * k);
    long *data_per_cluster = malloc(sizeof(long) * k);

    /*Select k unique random points in the data as starting centroids*/
    for(long p = 0; p < k; p++)
    {
        long row = 0;
        for(int q = 0; q < sizeof(long)/sizeof(unsigned char); q++)
        {
            unsigned char c = rand();
            row <<= (8 * sizeof(unsigned char));
            row |= c;
        }
        row = abs(row % src->rows);
        int unique = 0;
        while(!unique)
        {
            for(long q = 0; q < p; q++)
            {
                if(row == starting_indices[q])
                {
                    row = (row + 1) % src->rows;
                    break;
                }
            }
            unique = 1;
        }
        starting_indices[p] = row;
        for(long q = 0; q < src->cols; q++)
        {
            m_set(&old_centroids, p, q, m_get(src, row, q));
        }
    }
    int converged = 0;
    while(!converged)
    {
        converged = 1;

        for(long p = 0; p < k; p++)
        {
            data_per_cluster[p] = 0;
        }
        for(long p = 0; p < new_centroids.rows * new_centroids.cols; p++)
        {
            new_centroids.contents[p] = 0;
        }
        for(long p = 0; p < src->rows; p++)
        {
            double minimum_distance = -1;
            long closest_centroid = -1;
            for(long q = 0; q < k; q++)
            {
                double distance_from_centroid = 0;
                for(long r = 0; r < src->cols; r++)
                {
                    double diff = (src->contents[p * src->cols + r] - old_centroids.contents[q * old_centroids.cols + r]);
                    distance_from_centroid += diff*diff;
                }
                distance_from_centroid = sqrt(distance_from_centroid);
                if(distance_from_centroid < minimum_distance || minimum_distance < 0)
                {
                    minimum_distance = distance_from_centroid;
                    closest_centroid = q;
                }
            }
            if(closest_centroid != classifications[p])
            {
                converged = 0;
            }
            classifications[p] = closest_centroid;
            for(long q = 0; q < src->cols; q++)
            {
                new_centroids.contents[closest_centroid * new_centroids.cols + q] += src->contents[p * src->cols + q];
            }
            data_per_cluster[closest_centroid] += 1;
        }
        for(long p = 0; p < k; p++)
        {
            for(long q = 0; q < new_centroids.cols; q++)
            {
                m_set(&new_centroids, p, q, m_get(&new_centroids, p, q)/data_per_cluster[p]);
            }
        }
        
        Matrix temp = old_centroids;
        old_centroids = m_copy(&new_centroids);
        m_destroy(&temp);

    }
    free(starting_indices);
    free(classifications);
    return old_centroids;
}
long k_means_predict(Matrix *vector, Matrix *centroid_matrix)
{        
    double minimum_distance = -1;
    long closest_centroid;
    for(long i = 0; i < centroid_matrix->rows; i++)
    {
        double distance_from_centroid = 0;
        for(long j = 0; j < vector->rows * vector->cols; j++)
        {
            double diff = (centroid_matrix->contents[i * centroid_matrix->cols + j] - vector->contents[j]);
            distance_from_centroid += diff * diff;
        }
        distance_from_centroid = sqrt(distance_from_centroid);

        if(distance_from_centroid < minimum_distance || minimum_distance < 0)
        {
            minimum_distance = distance_from_centroid;
            closest_centroid = i;
        }
    }
    return closest_centroid;
}