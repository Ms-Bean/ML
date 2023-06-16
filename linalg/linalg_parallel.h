#include "linalg.h"
#include <pthread.h>
#define NUM_THREADS 16 /*Change this according to your machine's capabilities*/

pthread_mutex_t mutex;

struct mult_data
{
    Matrix *src1;
    Matrix *src2_transpose;
    Matrix *dst;
    long starting_index;
};
void *_m_mult_parallel(void *arg)
{
    struct mult_data *dat = (struct mult_data *)arg;

    Matrix *src1 = dat->src1;
    Matrix *src2_transpose = dat->src2_transpose;
    Matrix *dst = dat->dst;

    pthread_mutex_lock(&mutex);
        long starting_index = dat->starting_index++;
    pthread_mutex_unlock(&mutex);

    for(long i = starting_index; i < src1->rows; i += NUM_THREADS)
    {
        for(long j = 0; j < src2_transpose->rows; j++)
        {
            double sum = 0.0;
            for(long k = 0; k < src1->cols; k++)
            {
                sum += src1->contents[i * src1->cols + k] * src2_transpose->contents[j * src2_transpose->cols + k];
            } 
            dst->contents[i * dst->cols + j] = sum;
        }
    }
    return NULL;
}
Matrix m_mult_parallel(Matrix *src1, Matrix *src2)
{
    Matrix out = m_init(src1->rows, src2->cols);
    Matrix src2_transpose = m_transpose(src2);

    pthread_t *tids = malloc(sizeof(pthread_t) * NUM_THREADS);
    struct mult_data *dat = malloc(sizeof(struct mult_data));

    dat->src1 = src1;
    dat->src2_transpose = &src2_transpose;
    dat->dst = dst;
    dat->starting_index = 0;

    pthread_mutex_init(&mutex, NULL);
    for(long i = 0; i < NUM_THREADS; i++)
    {
        pthread_create(&tids[i], NULL, _m_mult_parallel, (void *)dat);
    }
    pthread_mutex_destroy(&mutex);

    for(long i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(tids[i], NULL);
    }

    m_destroy(&src2_transpose);
    free(dat);
    free(tids);
}
