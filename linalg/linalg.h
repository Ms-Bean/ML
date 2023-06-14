#include <stdlib.h>
#include <math.h>
#include <stdarg.h>
#include <string.h>
#include <stdio.h>


/*Misc functions*/
double dot_product(double *a, double *b, long len);
int is_equal(double a, double b);

typedef struct Matrix
{                       /*M00   M01    M02*/
    long rows;          /*M10   M11    M12*/
    long cols;          /*M20   M21    M22*/
    double *contents;  
} Matrix; 

/*Vector functions*/
double v_Lnorm(Matrix *src, long L);

/*Matrix creation and destruction functions*/
Matrix m_init(long rows, long cols);
Matrix m_create_transpose(Matrix *src);
Matrix m_create_identity_matrix(long I);
Matrix m_create_diag(Matrix *src);
void m_destroy(Matrix *dst);
void m_copy(Matrix *src, Matrix *dst);
void m_copy_c_matrix(void *src, Matrix *dst);

/*Visualization*/
void m_print(Matrix *src);
void m_label_print(Matrix *src, char *label);

/*Matrix manipulation*/
void m_transpose(Matrix *src, Matrix *dst);
void m_mult(Matrix *src1, Matrix *src2, Matrix *dst);
void m_scmult(double scalar, Matrix *dst);
void m_hadamard(Matrix *src1, Matrix *src2, Matrix *dst);
void m_swap_rows(Matrix *dst, long a, long b);
int m_inverse(Matrix *src, Matrix *dst); /*Returns true iff inverse was successfuly taken*/

/*Useful for solving systems of linear equations*/
void m_row_echelon(Matrix *dst); 
void m_reduced_row_echelon(Matrix *dst);
Matrix m_back_substitution(Matrix *src); /*Takes in augmented matrix in row-echelon, returns a vector containing the solution to the system if there is one*/

/*Matrix statistics*/
double m_trace(Matrix *src);
double m_frobenius_norm(Matrix *src);
int m_row_linear_independent(Matrix *src);
int m_column_linear_independent(Matrix *src);
long m_rank(Matrix *src);

typedef struct Tensor
{
    long rank;
    long *dims;
    
    double *contents;
} Tensor;
Tensor t_init(long rank, ...);
Tensor _t_init(long rank, long *dims);
long *_t_index_weights(Tensor *t1);;
long t_num_elements(Tensor *t1);
void _t_print_linear(Tensor *t1, long d_index, long l, long r, long *index_weights);
void t_print_linear(Tensor *t1);
double t_get(Tensor *src, ...);
void t_set(Tensor *dst, double val, ...);
void t_add(Tensor *t1, Tensor *t2, Tensor *dst);
void t_transpose(Tensor *src, Tensor *dst);
Tensor t_create_transpose(Tensor *src);
void t_copy_c_tensor(Tensor *dst, double *src, int num_elements);

double dot_product(double *a, double *b, long len)
{
    double sum;
    sum = 0;
    for(long i = 0; i < len; i++)
        sum += a[i] * b[i];
    return sum;
}
int is_equal(double a, double b)
{
    return fabs(a - b) <= 1e-9 * fabs(a) + fabs(b) + 1e-9; 
}
Matrix m_init(long rows, long cols)
{
    Matrix out;

    out.rows = rows;
    out.cols = cols;
    out.contents = calloc(rows*cols, sizeof(double));
    return out;
}
void m_destroy(Matrix *dst)
{
    free(dst->contents);
}
void m_copy(Matrix *src, Matrix *dst)
{
    for(long i = 0; i < src->cols * src->rows; i++)
    {
        dst->contents[i] = src->contents[i];
    }
}
double v_Lnorm(Matrix *src, long L)
{
    double sum;
    for(long i = 0; i < src->cols; i++)
    {
        sum += pow(src->contents[i], L);
    }
    return pow(sum, 1.0/L);
}
Matrix m_create_transition_matrix(Matrix *src1, Matrix *src2)
{
    Matrix out;
    Matrix augmented_matrix;
    Matrix solution;

    out = m_init(src1->rows, src1->cols);
    augmented_matrix = m_init(src1->rows, src1->cols + 1);
    for(long i = 0; i < src1->cols; i++)
    {
        for(long j = 0; j < src2->rows; j++)
        {
            long k;
            for(k = 0; k < src2->cols; k++)
            {
                augmented_matrix.contents[j * augmented_matrix.cols + k] = src2->contents[j * src2->cols + k];
            }
            augmented_matrix.contents[j * augmented_matrix.cols + k] = src1->contents[j * src1->cols + i];
        }
        m_row_echelon(&augmented_matrix);
        solution = m_back_substitution(&augmented_matrix);
        for(long j = 0; j < solution.rows; j++)
        {
            out.contents[j * out.cols + i] = solution.contents[j];
        }
    }
    m_destroy(&solution);
    m_destroy(&augmented_matrix);
    return out;
}
void m_print(Matrix *src)
{
    for(long i = 0; i < src->rows; i++)
    {
        printf("[");
        for(long j = 0; j < src->cols; j++)
        {
            printf("%8.3lf", src->contents[i * src->cols + j]);
            if(j < src->cols-1)
                printf(", ");
        }
        printf("]\n");
    }
    printf("\n");
}
void m_label_print(Matrix *src, char *label)
{
    printf("%s\n", label);
    for(long i = 0; i < src->rows; i++)
    {
        printf("[");
        for(long j = 0; j < src->cols; j++)
        {
            printf("%0.3lf", src->contents[i * src->cols + j]);
            if(j < src->cols-1)
                printf(", ");
        }
        printf("]\n");
    }
    printf("\n");
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
    dst = m_init(src->cols, src-> rows);
    m_transpose(src, &dst);

    return dst;
}
void m_mult(Matrix *src1, Matrix *src2, Matrix *dst)
{
    Matrix src2_transpose = m_create_transpose(src2);
    for(long i = 0; i < src1->rows; i++)
        for(long j = 0; j < src2_transpose.rows; j++)
            dst->contents[i * dst->cols + j] = dot_product(src1->contents + (i * src1->cols), src2_transpose.contents + (j * src2_transpose.cols), src1->cols);
    m_destroy(&src2_transpose);
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
Matrix m_create_diag(Matrix *src)
{
    Matrix out = m_init(src->cols, src->cols);
    for(long i = 0; i < src->cols; i++)
        out.contents[i * (src->cols + 1)] = src->contents[i];

    return out;
}
void m_swap_rows(Matrix *dst, long a, long b)
{
    double temp;
    for(long i = 0; i < dst->cols; i++)
    {
        temp = dst->contents[a * dst->cols + i];
        dst->contents[a * dst->cols + i] = dst->contents[b * dst->cols + i];
        dst->contents[b * dst->cols + i] = temp;
    }
}
void m_row_echelon(Matrix *dst)
{
    long h = 0;
    long k = 0;

    while(h < dst->rows && k < dst->cols)
    {
        for(long m = h; m < dst->rows; m++)
        {
            if(!is_equal(dst->contents[m * dst->cols + k], 0.0))
            {
                m_swap_rows(dst, m, h);
                for(long i = h + 1; i < dst->rows; i++)
                {
                    double weight = dst->contents[i * dst->cols + k] / dst->contents[h * dst->cols + k];
                    dst->contents[i * dst->cols + k] = 0.0;
                    for(long j = k + 1; j < dst->cols; j++)
                    {
                        dst->contents[i * dst->cols + j] -= dst->contents[h * dst->cols + j] * weight;
                    }
                }
                h++;
                k++;
                continue;
            }
        }
        k++;
    }
}
void m_reduced_row_echelon(Matrix *dst)
{
    for(long i = 0; i < dst->rows; i++)
    {
        for(long j = 0; j < dst->cols; j++)
        {
            long k;
            double weight;
            for(k = i; k < dst->rows; k++)
            {
                if(!is_equal(dst->contents[k * dst->cols + j], 0.0))
                {
                    if(k == i)
                    {
                        break;
                    }
                    m_swap_rows(dst, k, i);
                    break;
                }
            }
            if(k == dst->rows)
            {
                continue;
            }
            weight = 1.0/dst->contents[i * dst->cols + j];
            for(k = 0; k < dst->cols; k++)
            {
                dst->contents[i * dst->cols + k] *= weight;
            }
            for(k = 0; k < dst->rows; k++)
            {
                if(k == i)
                    continue;
                if(!is_equal(dst->contents[k * dst->cols + j], 0.0))
                {
                    weight = -dst->contents[k * dst->cols + j];
                    for(long l = 0; l < dst->cols; l++)
                    {
                        dst->contents[k * dst->cols + l] += dst->contents[i * dst->cols + l] * weight;
                    }
                }
            }
            i++;
        }
    }
}
Matrix m_back_substitution(Matrix *src)
{
    Matrix out;
    out.rows = out.cols = 0;
    if(src->cols > src->rows + 1)
    {
        return out;
    }

    for(long i = 0; i < src->cols - 1; i++)
        if(is_equal(src->contents[i * src->cols + i], 0.0))
            return out;

    for(long i = 0; i < src->rows; i++)
        for(long j = 0; j < i; j++)
            if(!is_equal(src->contents[i * src->cols + j], 0.0))
                return out;

    out = m_init(src->cols-1, 1);
    double sum = 0;
    for(long i = src->cols - 2; i >= 0; i--)
    {
        sum = src->contents[i * src->cols + src->cols - 1];
        for(long j = i + 1; j < src->cols-1; j++)
        {
            sum -= src->contents[i * src->cols + j] * out.contents[j];
        }
        out.contents[i] = sum / src->contents[i * src->cols + i];
    }
    return out;
}
double m_determinant(Matrix *src)
{
    Matrix triangle = m_init(src->rows, src->cols);

    m_copy(src, &triangle);
    m_row_echelon(&triangle);
    double product = 1.0;
    for(long i = 0; i < src->rows; i++)
    {
        product *= triangle.contents[i * (triangle.cols + 1)];
    }
    return product;
}
void m_qr_factorization(Matrix *src, Matrix *q, Matrix *r) /*Modified gram-schmidt*/
{
    *q = m_init(src->rows, src->cols);
    m_copy(src, q);
    
    *r = m_init(src->rows, src->cols);

    for(long i = 0; i < src->rows; i++)
    {
        double s = 0;
        for(long j = 0; j < i; j++)
        {
            double s = 0.0;
            for(long k = 0; k < q->rows; k++)
            {
                s += q->contents[k * q->cols + i] * q->contents[k * q->cols + j];
            }
            r->contents[j * q->cols + i] = s;
            for(long k = 0; k < q->rows; k++)
            {
                q->contents[k * q->cols + i] -= s * q->contents[k * q->cols + j];
            }
        }
        double norm = 0;
        for(long j = 0; j < q->rows; j++)
        {
            norm += q->contents[j * q->cols + i]*q->contents[j * q->cols + i];
        }
        norm = sqrt(norm);
        r->contents[i * (r->cols + 1)] = norm;
        for(long j = 0; j < q->cols; j++)
        {
            q->contents[j * q->cols + i] /= norm;
        }
    }
}
Matrix m_eigenvalues(Matrix *src, long iterations)
{
    Matrix q, r;
    Matrix a = m_init(src->rows, src->cols);
    Matrix out = m_init(src->rows, 1);

    m_copy(src, &a);
    for(long i = 0; i < iterations; i++)
    {
        m_qr_factorization(&a, &q, &r);
        m_mult(&r, &q, &a);
        m_destroy(&q);
        m_destroy(&r);
    }
    for(long i = 0; i < a.rows; i++)
    {
        out.contents[i] = a.contents[i * (a.cols + 1)];
    }
    m_destroy(&a);
    return out;
}
double m_trace(Matrix *src)
{
    double sum = 0;
    for(long i = 0; i < src->rows; i++)
        sum += src->contents[i * (src->cols + 1)];

    return sum;
}
double m_frobenius_norm(Matrix *src)
{
    double sum = 0;
    for(long i = 0; i < src->rows * src->cols; i++)
        sum += (src->contents[i]*src->contents[i]);       

    return sqrt(sum);
}
int m_inverse(Matrix *src, Matrix *dst)
{
    Matrix temp = m_init(dst->rows, dst->cols*2);
    for(long i = 0; i < temp.rows; i++)
    {
        temp.contents[i * (temp.cols + 1) + temp.rows] = 1.0;
    }
    for(long i = 0; i < dst->rows; i++)
        for(long j = 0; j < dst->cols; j++)
            temp.contents[i * temp.cols + j] = src->contents[i * dst->cols + j];
    
    m_reduced_row_echelon(&temp);

    for(long i = 0; i < temp.rows; i++)
    {
        for(long j = 0; j < temp.rows; j++)
        {
            if(i == j && !is_equal(temp.contents[2 * i * temp.cols + j], 1.0))
            {
                m_destroy(&temp);
                return 0;
            }
            else if(i != j && !is_equal(temp.contents[2 * i * temp.cols + j], 0.0))
            {
                m_destroy(&temp);
                return 0;
            }
        }
    } 
    for(long i = 0; i < dst->rows; i++)
        for(long j = 0; j < dst->cols; j++)
            dst->contents[i * dst->cols + j] = temp.contents[i * temp.cols + temp.rows + j];
    
    m_destroy(&temp);
    return 1;
}
int m_column_linear_independent(Matrix *src)
{
    Matrix temp;

    if(src->cols > src->rows)
        return 0;

    temp = m_init(src->rows, src->cols);
    m_copy(src, &temp);
    m_row_echelon(&temp);

    for(long i = 0; i < temp.cols; i++)
    {
        if(is_equal(temp.contents[i * temp.cols + i], 0.0))
        {
            m_destroy(&temp);
            return 0;
        }
    }
    m_destroy(&temp);
    return 1;
}
int m_row_linear_independent(Matrix *src)
{
    Matrix temp;

    if(src->cols > src->rows)
        return 0;

    temp = m_create_transpose(src);
    m_row_echelon(&temp);

    for(long i = 0; i < temp.cols; i++)
    {
        if(is_equal(temp.contents[i * temp.cols + i], 0.0))
        {
            m_destroy(&temp);
            return 0;
        }
    }
    m_destroy(&temp);
    return 1;
}
long m_rank(Matrix *src)
{
    Matrix temp;
    long rank = 0;

    temp = m_create_transpose(src);
    m_row_echelon(&temp);

    long i = 0;
    long j = 0;
    while(j < temp.cols && i < temp.rows)
    {
        if(!is_equal(temp.contents[i * temp.cols + j], 0.0))
        {
            rank++;
            i++;
        }
        j++;
    }
    m_destroy(&temp);
    return rank;
}

Tensor _t_init(long rank, long *dims)
{
    Tensor out;
    double num_elements = 1;

    out.rank = rank;
    out.dims = malloc(sizeof(long) * rank);

    for(long i = 0; i < rank; i++)
    {
        out.dims[i] = dims[i];
        num_elements *= out.dims[i];
    }
    out.contents = calloc(num_elements, sizeof(double));
    return out;
}
Tensor t_init(long rank, ...)
{
    Tensor out;
    va_list list_ptr;
    long *dims = malloc(sizeof(long) * rank);

    va_start(list_ptr, rank);
    for(long i = 0; i < rank; i++)
        dims[i] = va_arg(list_ptr, long);
    va_end(list_ptr);
    out = _t_init(rank, dims);
    free(dims);
    return out;
}
void t_copy_c_tensor(Tensor *dst, double *src, int num_elements)
{
    for(long i = 0; i < num_elements; i++)
        dst->contents[i] = src[i];
}
long t_num_elements(Tensor *t1)
{
    long i;
    long element_count;
    
    element_count = 1;
    for(i = 0; i < t1->rank; i++)
        element_count *= t1->dims[i];

    return element_count;
}
long *_t_index_weights(Tensor *t1)
{
    long *out = calloc(t1->rank, sizeof(long));
    for(long i = 0; i < t1->rank; i++)
    {
        long product = 1;
        for(long j = i + 1; j < t1->rank; j++)
            product *= t1->dims[j];
        out[i] = product;
    }
    return out;
}
void _t_print_linear(Tensor *t1, long d_index, long l, long r, long *index_weights)
{
    if(d_index >= t1->rank-1)
    {
        printf("[");
        for(long i = l; i < r; i++)
        {
            printf("%lf", t1->contents[i]);
            if(i < r-1)
                printf(", ");
        }
        printf("]");
        return;
    }
    else
    {
        printf("[");
        for(long i = l; i < r; i += index_weights[d_index])
        {
            _t_print_linear(t1, d_index+1, i, i+index_weights[d_index], index_weights);
        }
        printf("]");
    }
}
void t_print_linear(Tensor *t1)
{
    long *index_weights = _t_index_weights(t1);
    _t_print_linear(t1, 0, 0, t_num_elements(t1), index_weights);
    printf("\n");
    free(index_weights);
}
double t_get(Tensor *src, ...)
{
    va_list list_ptr;
    long get_index;

    get_index = 0;

    va_start(list_ptr, src);
    for(long i = 0; i < src->rank; i++)
        get_index += va_arg(list_ptr, long) * src->dims[i];
    va_end(list_ptr);

    return src->contents[get_index];
}
void t_set(Tensor *dst, double val, ...)
{
    va_list list_ptr;
    long set_index;

    set_index = 0;

    va_start(list_ptr, val);
    for(long i = 0; i < dst->rank; i++)
        set_index += va_arg(list_ptr, long) * dst->dims[i];
    va_end(list_ptr);

    dst->contents[set_index] = val;
}
void t_add(Tensor *t1, Tensor *t2, Tensor *dst)
{
    for(long i = 0; i < t_num_elements(t1); i++)
        dst->contents[i] = t2->contents[i] + t1->contents[i];
}
void t_transpose(Tensor *src, Tensor *dst)
{
    long *indices = calloc(src->rank, sizeof(long));
    long *index_weights = _t_index_weights(dst);

    for(long i = 0; i < t_num_elements(src); i++)
    {
        long set_index = 0;
        for(long j = 0; j < dst->rank; j++)
            set_index += indices[j] * index_weights[dst->rank-j-1];
        dst->contents[set_index] = src->contents[i];

        for(long j = src->rank-1; j >= 0; j--)
        {
            if(src->dims[j] > indices[j]+1)
            {
                indices[j]++;
                memset((indices + j + 1), 0, sizeof(long)*(src->rank - j - 1));
                break;
            }
        }
    }
    free(indices);
    free(index_weights);
}
Tensor t_create_transpose(Tensor *src)
{
    Tensor out;

    long *reverse_dims = malloc(sizeof(long) * src->rank);
    for(long i = 0; i < src->rank; i++)
        reverse_dims[i] = src->dims[src->rank-i-1];

    out = _t_init(src->rank, reverse_dims);     
    t_transpose(src, &out);
    return out;
}
void t_destroy(Tensor *t1)
{
    free(t1->contents);
    free(t1->dims);
}