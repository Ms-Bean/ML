#include <stdlib.h>
typedef struct Matrix
{
    long rows;
    long cols;
    double *contents;  
} Matrix;

Matrix m_init(long rows, long cols)
{
    Matrix out;

    out.rows = rows;
    out.cols = cols;
    out.contents = calloc(rows*cols, sizeof(double));
    return out;
}
void destroy(Matrix *dst)
{
    free(dst->contents);
}
void m_print(Matrix *src)
{
    for(long i = 0; i < src->rows; i++)
    {
        printf("[");
        for(long j = 0; j < src->cols; j++)
        {
            printf("%lf", src->contents[i * src->cols + j]);
            if(j < src->cols-1)
                printf(", ");
        }
        printf("]\n");
    }
}
void m_copy_c_matrix(void *src, Matrix *dst)
{
    double *c_matrix = (double *)src;

    for(long i = 0; i < dst->cols * dst->rows; i++)
        dst->contents[i] = c_matrix[i];
}
void m_transpose(Matrix *src, Matrix *dst)
{
    for(long i = 0; i < dst->rows; i++)
        for(long j = 0; j < dst->cols; j++)
            dst->contents[i * dst->cols + j] = src->contents[j * src->cols + i];
}
Matrix m_create_transpose(Matrix *src)
{
    Matrix dst;

    dst.cols = src->rows;
    dst.rows = src->cols;
    dst.contents = calloc(src->rows * src->cols, sizeof(double));

    m_transpose(src, &dst);
    return dst;
}
double dot_product(double *a, double *b, long len)
{
    double sum;
    sum = 0;
    for(long i = 0; i < len; i++)
        sum += a[i] * b[i];
    return sum;
}
void m_mult(Matrix *src1, Matrix *src2, Matrix *dst)
{
    Matrix src2_transpose = m_create_transpose(src2);
    for(long i = 0; i < src1->rows; i++)
        for(long j = 0; j < src2_transpose.rows; j++)
            dst->contents[i * dst->cols + j] = dot_product(src1->contents + (i * src1->cols), src2_transpose.contents + (j * src2_transpose.cols), src1->cols);
    destroy(&src2_transpose);
}
void m_scmult(double scalar, Matrix *dst)
{
    for(long i = 0; i < dst->cols * dst->rows; i++)
        dst->contents[i] *= scalar;
}
void m_hadamard(Matrix *src1, Matrix *src2, Matrix *dst)
{
    for(long i = 0; i < src1->rows * src1->cols; i++)
        dst->contents[i] = src1->contents[i] * src2->contents[i];
}
Matrix m_create_identity_matrix(long I)
{
    Matrix out = m_init(I, I);

    for(long i = 0; i < I; i++)
        out.contents[i * (I+1)] = 1;
        
    return out;
}