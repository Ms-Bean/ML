#include "../c_ml.h"
int main(int argc, char **argv)
{
    if(argc < 2)
    {
        printf("No filename given\n");
        exit(2);
    }

    Matrix data = m_read_csv(argv[1]);
    m_standardize(&data);
    Matrix proj;
    Matrix reduced = m_PCA_dimensionality_reduction(&data, &proj, 3);
    Matrix centroid_matrix = m_k_means_clustering(&reduced, 10, 3);

    Matrix column_vector = m_init(3, 1);

    FILE *cluster_1 = fopen("cluster_1.csv", "w");
    FILE *cluster_2 = fopen("cluster_2.csv", "w");
    FILE *cluster_3 = fopen("cluster_3.csv", "w");

    fprintf(cluster_1, "x,y,z\n");
    fprintf(cluster_2, "x,y,z\n");
    fprintf(cluster_3, "x,y,z\n");

    FILE *files[3];
    files[0] = cluster_1;
    files[1] = cluster_2;
    files[2] = cluster_3;

    for(long i = 0; i < reduced.rows; i++)
    {
        for(long j = 0; j < 3; j++)
        {
            column_vector.contents[j] = m_get(&reduced, i, j);
        }
        long cluster = k_means_predict(&column_vector, &centroid_matrix);
        fprintf(files[cluster], "%lf,%lf,%lf\n", m_get(&reduced, i, 0), m_get(&reduced, i, 1), m_get(&reduced, i, 2)); 
    }

    fclose(cluster_1);
    fclose(cluster_2);
    fclose(cluster_3);
    return 0;
}