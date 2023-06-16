#include <stdlib.h>
#include <math.h>
#include <stdarg.h>
#include <string.h>
#include <stdio.h>

/*Misc functions*/
double dot_product(double *a, double *b, long len);
int is_equal(double a, double b);
int is_equal_e(double a, double b, double epsilon1, double epsilon2);
void quicksort_descending(double *arr, long low, long high);

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
Matrix m_read_csv(char *filename);
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
int m_inverse(Matrix *src, Matrix *dst); 
Matrix m_create_transpose(Matrix *src);
Matrix m_create_identity_matrix(long I);
Matrix m_create_diag(Matrix *src);

/*Useful for solving systems of linear equations*/
void m_row_echelon(Matrix *dst); 
void m_reduced_row_echelon(Matrix *dst);
Matrix m_back_substitution(Matrix *src); 

/*Matrix statistics*/
double m_trace(Matrix *src);
double m_frobenius_norm(Matrix *src);
int m_row_linear_independent(Matrix *src);
int m_column_linear_independent(Matrix *src);
long m_rank(Matrix *src);

/*Data matrix manipulation*/
Matrix m_create_covariance_matrix(Matrix *src);

/*Machine learning algorithms*/
Matrix m_PCA_dimensionality_reduction(Matrix *src, Matrix *proj, long k);

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
int is_equal_e(double a, double b, double epsilon1, double epsilon2)
{
    return fabs(a - b) <= epsilon1 * fabs(a) + fabs(b) + epsilon2; 
}

void quicksort_descending(double *arr, long low, long high)
{
	long i, j;
	double pivot, temp;
	if(low >= high)
		return;
	pivot = arr[high];
	for(j = i = low; j <= high - 1; j++)
	{
		if(arr[j] >= pivot)
		{
			temp = arr[i];
			arr[i] = arr[j];
			arr[j] = temp;
			i++;
		}
	}
	temp = arr[i];
	arr[i] = arr[j];
	arr[j] = temp;
	quicksort_descending(arr, low, i-1);
	quicksort_descending(arr, i+1, high);
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
double m_get(Matrix *src, long row, long col)
{
    return src->contents[row * src->cols + col];
}
double v_Lnorm(Matrix *src, long L)
{
    double sum;
    for(long i = 0; i < src->cols*src->rows; i++)
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
            printf("%8.5lf", src->contents[i * src->cols + j]);
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
            printf("%8.5lf", src->contents[i * src->cols + j]);
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
                if(m != h)
                {
                    m_swap_rows(dst, m, h);
                }
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
long m_row_echelon_count_swaps(Matrix *dst)
{
    long h = 0;
    long k = 0;
    long swaps = 0;
    while(h < dst->rows && k < dst->cols)
    {
        for(long m = h; m < dst->rows; m++)
        {
            if(!is_equal(dst->contents[m * dst->cols + k], 0.0))
            {
                if(m != h)
                {
                    m_swap_rows(dst, m, h);
                    swaps++;
                }
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
    return swaps;
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
/*Takes in an augmented matrix in REF, returns a column vector of solutions to the system of linear equations*/
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
    long swaps = m_row_echelon_count_swaps(&triangle);
    double product = 1.0;
    for(long i = 0; i < src->rows; i++)
    {
        product *= triangle.contents[i * (triangle.cols + 1)];
    }
    if(swaps % 2 == 0)
        return product;
    return -product;
}
/*Cholesky-banachiewicz algorithm, returns lower triangular matrix L where A=LL(transpose)*/
Matrix m_cholesky_factor(Matrix *src)
{
    Matrix out = m_init(src->rows, src->cols);

    for(long i = 0; i < src->rows; i++)
    {
        for(long j = 0; j <= i; j++)
        {
            double sum = 0;
            for(long k = 0; k < j; k++)
            {
                sum += out.contents[i * out.cols + k] * out.contents[j * out.cols + k];
            }

            if(j == i)
            {
                out.contents[i * out.cols + j] = sqrt(src->contents[i * (src->cols + 1)] - sum);
            }
            else
            {
                out.contents[i * out.cols + j] = 1.0/out.contents[j * (out.cols + 1)] * (src->contents[i * src->cols + j] - sum);
            }
        }
    }
    return out;
}
/*Returns a column vector containing the mean of each variable in a data matrix*/
Matrix m_mean(Matrix *src)
{
    Matrix out = m_init(src->cols, 1);
    for(long i = 0; i < src->rows; i++)
    {
        for(long j = 0; j < src->cols; j++)
        {
            out.contents[j] += src->contents[i * src->cols + j];
        }
    }
    for(long i = 0; i < out.rows; i++)
    {
        out.contents[i] /= src->rows;
    }
    return out;
}
/*Returns a column vector containing the standard deviation of each variable in a data matrix*/
Matrix m_stdev(Matrix *src)
{
    Matrix means = m_mean(src);
    Matrix out = m_init(src->cols, 1);

    for(long i = 0; i < src->rows; i++)
    {
        for(long j = 0; j < src->cols; j++)
        {
            out.contents[j] += (means.contents[j] - src->contents[i * src->cols + j])*(means.contents[j] - src->contents[i * src->cols + j]);
        }
    }
    for(long i = 0; i < out.rows; i++)
    {
        out.contents[i] = sqrt(out.contents[i] / (src->rows-1));
    }
    m_destroy(&means);
    return out;
}
/*Takes in a data matrix, performs Z-score normalization*/
void m_standardize(Matrix *dst)
{
    Matrix means = m_mean(dst);
    Matrix stdevs = m_stdev(dst);
    for(long i = 0; i < dst->rows; i++)
    {
        for(long j = 0; j < dst->cols; j++)
        {
            dst->contents[i * dst->cols + j] = (dst->contents[i * dst->cols + j] - means.contents[j]) / stdevs.contents[j];
        }
    }
    m_destroy(&means);
    m_destroy(&stdevs);
}
/*q and r should be pointers to uninitialized matrices*/
void m_qr_factorization(Matrix *src, Matrix *q, Matrix *r) 
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
 /*Performs iterations iterations of QR algorithm, returns a column vector of eigenvalues*/
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
/*Returns a matrix with eigenvectors listed as columns. Eigenvectors will have L2 norms equal to 1, and will be in descending order by eigenvalue*/
Matrix m_eigenvectors(Matrix *src)
{
    Matrix out;
    out = m_init(src->rows, src->cols);

    Matrix eigenvalues = m_eigenvalues(src, 100);
    quicksort_descending(eigenvalues.contents, 0, eigenvalues.rows-1);

    Matrix augmented_matrix = m_init(src->rows, src->cols + 1);
    for(long i = 0; i < eigenvalues.rows; i++)
    {
        for(long j = 0; j < src->rows; j++)
        {
            long k;
            for(k = 0; k < src->cols; k++)
            {
                augmented_matrix.contents[augmented_matrix.cols * j + k] = src->contents[src->cols * j + k];
            }
            augmented_matrix.contents[augmented_matrix.cols * j + k] = 0;
        }
        for(long j = 0; j < src->rows; j++)
        {
            augmented_matrix.contents[j * (augmented_matrix.cols + 1)] -= eigenvalues.contents[i];
        }

        m_row_echelon(&augmented_matrix);

        if(!is_equal_e(augmented_matrix.contents[(augmented_matrix.rows - 1) * augmented_matrix.cols + augmented_matrix.cols - 1], 0.0, 1e-4, 1e-4) || !is_equal_e(augmented_matrix.contents[(augmented_matrix.rows - 1) * augmented_matrix.cols + augmented_matrix.cols - 2], 0.0, 1e-4, 1e-4))
        {
            printf("%0.16lf\n", augmented_matrix.contents[(augmented_matrix.rows - 1) * augmented_matrix.cols + augmented_matrix.cols - 1]);
            printf("Eigenvector computation failed for eigenvalue %0.16lf\n", eigenvalues.contents[i]);
        }
        augmented_matrix.contents[(augmented_matrix.rows - 1) * augmented_matrix.cols + augmented_matrix.cols - 1] = 1;
        augmented_matrix.contents[(augmented_matrix.rows - 1) * augmented_matrix.cols + augmented_matrix.cols - 2] = 1;
        Matrix solution = m_back_substitution(&augmented_matrix);

        if(solution.rows != 0)
        {
            double weight = 1.0/v_Lnorm(&solution, 2);
            m_scmult(weight, &solution);
            for(long j = 0; j < solution.rows; j++)
            {
                out.contents[j * out.cols + i] = solution.contents[j];
            }
        }
        m_destroy(&solution);
    }
    m_destroy(&augmented_matrix);
    m_destroy(&eigenvalues);

    return out;
}
/*Takes in a data matrix, hopefully a standardized one, and reduces to k dimensional dataset. Compute v@proj to project column vector v onto the lower-dimensional plane*/
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

    Matrix new = m_init(src->rows, k);
    m_mult(src, proj, &new);

    return new;    
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
 /*Calculates inverse if there is one, returns false iff there is none.*/
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
/*Returns true iff all columns in matrix are linearly independent*/
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
 /*Returns true iff all rows in matrix are linearly independent*/
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
 /*Reads a CSV file of floating point values, converts to data matrix*/
Matrix m_read_csv(char *filename)
{
    FILE *fp;
    Matrix out;

    fp = fopen(filename, "r");

    long column_count = 0;
    long row_count = 0;

    int c;
    while((c = getc(fp)) != '\n' && c != EOF);
    while((c = getc(fp)) != EOF)
    {
        if((c == ',' || c == '\n') && row_count == 0)
        {
            column_count++;
        }
        if(c == '\n')
        {
            row_count++;
        }
    }
    fclose(fp);
    out = m_init(row_count, column_count);

    long int_val = 0;
    double decimal_val = 0.0;
    int is_negative = 0;
    double divisor = 1.0;

    long row = 0;
    long column = 0;

    fp = fopen(filename, "r");
    while((c = getc(fp)) != '\n' && c != EOF);
    while((c = getc(fp)) != EOF)
    {
        if(c == '-')
        {
            is_negative = 1;
        }
        else if(c == ',' || c == '\n')
        {
            out.contents[row * out.cols + column] = is_negative ? -decimal_val - int_val : decimal_val + int_val;

            int_val = decimal_val = 0;
            divisor = 1;
            is_negative = 0;

            if(c == ',')
                column++;
            else if(c == '\n')
            {
                column = 0;
                row++;
            }
        }
        else if(divisor == 1)
        {
            if(c == '.')
            {
                divisor *= 10;
            }
            else
            {
                int_val = int_val * 10 + c - '0';
            }
        }
        else
        {
            decimal_val += (double)(c - '0')/divisor;
            divisor *= 10;
        }
    }
    fclose(fp);
    return out;
}
/*Takes in a data matrix, creates a covariance matrix*/
Matrix m_create_covariance_matrix(Matrix *src)
{
    Matrix out = m_init(src->cols, src->cols);

    for(long i = 0; i < out.rows; i++)
    {
        for(long j = i; j < out.cols; j++)
        {
            double mean_i, mean_j;
            mean_i = mean_j = 0;
            for(long k = 0; k < src->rows; k++)
            {
                mean_i += src->contents[k * src->cols + i];
                mean_j += src->contents[k * src->cols + j];
            }
            mean_i /= src->rows;
            mean_j /= src->rows;

            double covar = 0;
            for(long k = 0; k < src->rows; k++)
            {
               covar += (src->contents[k * src->cols + i] - mean_i)*(src->contents[k * src->cols + j] - mean_j);
            }
            covar /= src->rows;
            out.contents[i * out.cols + j] = covar;
        }
    }
    for(long i = 0; i < out.rows; i++)
    {
        for(long j = 0; j < i; j++)
        {
            out.contents[i * out.cols + j] = out.contents[j * out.cols + i];
        }
    }
    return out;
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