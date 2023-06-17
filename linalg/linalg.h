#include <stdlib.h>
#include <math.h>
#include <stdarg.h>
#include <string.h>
#include <stdio.h>

typedef struct Matrix
{                       /*M00   M01    M02*/
    long rows;          /*M10   M11    M12*/
    long cols;          /*M20   M21    M22*/
    double *contents;  
} Matrix; 

/*Misc functions*/
double dot_product(double *a, double *b, long len);
int is_equal(double a, double b);
int is_equal_e(double a, double b, double epsilon1, double epsilon2);
void quicksort_descending(double *arr, long low, long high);


Matrix m_init(long rows, long cols);
Matrix m_copy(Matrix *src);
Matrix m_copy_c_matrix(void *src, long rows, long cols);
Matrix m_read_csv(char *filename);
void m_destroy(Matrix *dst);

double m_get(Matrix *src, long row, long col);
void m_set(Matrix *dst, long row, long col, double val);

double v_Lnorm(Matrix *src, long L);
double v_L2norm(Matrix *src, long L);
double m_determinant(Matrix *src);
double m_trace(Matrix *src);
double m_frobenius_norm(Matrix *src);
int m_column_linear_independent(Matrix *src);
int m_row_linear_independent(Matrix *src);
long m_rank(Matrix *src);

void m_print(Matrix *src);
void m_label_print(Matrix *src, char *label);

Matrix m_transpose(Matrix *src);
Matrix m_mult(Matrix *src1, Matrix *src2);
Matrix m_add(Matrix *src1, Matrix *src2);
Matrix m_sub(Matrix *src1, Matrix *src2);
Matrix m_hadamard(Matrix *src1, Matrix *src2);
Matrix m_identity_matrix(long I);
Matrix m_diag(Matrix *src);
Matrix m_cholesky_factor(Matrix *src);
Matrix m_inverse(Matrix *src);

void m_scmult(double scalar, Matrix *dst);
void m_swap_rows(Matrix *dst, long a, long b);
void m_row_echelon(Matrix *dst);
long m_row_echelon_count_swaps(Matrix *dst);
void m_reduced_row_echelon(Matrix *dst);
void m_standardize(Matrix *dst);

Matrix m_mean(Matrix *src);
Matrix m_stdev(Matrix *src);
Matrix m_create_covariance_matrix(Matrix *src);
Matrix m_back_substitution(Matrix *src);
void m_qr_factorization(Matrix *src, Matrix *q, Matrix *r);
Matrix m_eigenvalues(Matrix *src, long iterations);
Matrix m_eigenvectors(Matrix *src, long iterations);


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
Matrix m_copy(Matrix *src)
{
    Matrix out = m_init(src->rows, src->cols);
    for(long i = 0; i < src->cols * src->rows; i++)
    {
        out.contents[i] = src->contents[i];
    }
    return out;
}
double m_get(Matrix *src, long row, long col)
{
    return src->contents[row * src->cols + col];
}
void m_set(Matrix *dst, long row, long col, double val)
{
    dst->contents[row *dst->cols + col] = val;
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
double v_L2norm(Matrix *src, long L)
{    
    double sum;
    for(long i = 0; i < src->cols*src->rows; i++)
    {
        sum += pow(src->contents[i], L);
    }
    return(sqrt(sum));
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
Matrix m_copy_c_matrix(void *src, long rows, long cols)
{
    Matrix out = m_init(rows, cols);
    double *c_matrix = (double *)src;

    for(long i = 0; i < out.cols * out.rows; i++)
        out.contents[i] = c_matrix[i];
    return out;
}
Matrix m_transpose(Matrix *src)
{
    Matrix out;
    out = m_init(src->cols, src->rows);
    for(long i = 0; i < out.rows; i++)
        for(long j = 0; j < out.cols; j++)
            out.contents[i * out.cols + j] = src->contents[j * src->cols + i];

    return out;
}
/*Returns src1+src2*/
Matrix m_add(Matrix *src1, Matrix *src2)
{
    Matrix out = m_init(src1->rows, src1->cols);

    for(long i = 0; i < src1->rows * src1->cols; i++)
    {
        out.contents[i] = src1->contents[i] + src2->contents[i];
    }
    return out;
}
/*Returns src1-src2*/
Matrix m_sub(Matrix *src1, Matrix *src2)
{

    Matrix out = m_init(src1->rows, src1->cols);

    for(long i = 0; i < src1->rows * src1->cols; i++)
    {
        out.contents[i] = src1->contents[i] - src2->contents[i];
    }
    return out;
}
/*Returns matrix containing src1@src2*/
Matrix m_mult(Matrix *src1, Matrix *src2)
{
    Matrix out = m_init(src1->rows, src2->cols);
    Matrix src2_transpose = m_transpose(src2);
    for(long i = 0; i < src1->rows; i++)
        for(long j = 0; j < src2_transpose.rows; j++)
            out.contents[i * out.cols + j] = dot_product(src1->contents + (i * src1->cols), src2_transpose.contents + (j * src2_transpose.cols), src1->cols);
    m_destroy(&src2_transpose);
    return out;
}
/*Scales dst by scalar*/
void m_scmult(double scalar, Matrix *dst)
{
    for(long i = 0; i < dst->cols * dst->rows; i++)
        dst->contents[i] *= scalar;
}
/*Returns matrix containing hadamard product of src1 and src2*/
Matrix m_hadamard(Matrix *src1, Matrix *src2)
{
    Matrix out = m_init(src1->rows, src1->cols);
    for(long i = 0; i < src1->rows * src1->cols; i++)
        out.contents[i] = src1->contents[i] * src2->contents[i];
    return out;
}
Matrix m_identity_matrix(long I)
{
    Matrix out = m_init(I, I);

    for(long i = 0; i < I; i++)
        out.contents[i * (I+1)] = 1;
        
    return out;
}
Matrix m_diag(Matrix *src)
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
    Matrix triangle = m_copy(src);

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
    *q = m_copy(src);
    *r = m_init(src->rows, src->cols);

    for(long i = 0; i < src->rows; i++)
    {
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
    Matrix a = m_copy(src);
    Matrix out = m_init(src->rows, 1);

    for(long i = 0; i < iterations; i++)
    {
        Matrix temp = a;

        m_qr_factorization(&a, &q, &r);
        a = m_mult(&r, &q);
        m_destroy(&temp);
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
Matrix m_eigenvectors(Matrix *src, long iterations)
{
    Matrix q, r;
    Matrix out = m_identity_matrix(src->rows);
    Matrix a = m_copy(src);
    for(long i = 0; i < iterations; i++)
    {
        Matrix temp = a;

        m_qr_factorization(&a, &q, &r);
        a = m_mult(&r, &q);
        Matrix eigenvectors_new = m_mult(&out, &q);
        for(long i = 0; i < eigenvectors_new.rows * eigenvectors_new.cols; i++)
        {
            out.contents[i] = eigenvectors_new.contents[i];
        }
        m_destroy(&eigenvectors_new);
        m_destroy(&temp);
        m_destroy(&q);
        m_destroy(&r);
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
 /*Calculates inverse if there is one, otherwise no memory will be allocated and rows and cols will equal 0*/
Matrix m_inverse(Matrix *src)
{
    Matrix out;
    Matrix temp = m_init(src->rows, src->cols*2);
    out.rows = out.cols = 0;
    
    for(long i = 0; i < temp.rows; i++)
    {
        temp.contents[i * (temp.cols + 1) + temp.rows] = 1.0;
    }
    for(long i = 0; i < src->rows; i++)
        for(long j = 0; j < src->cols; j++)
            temp.contents[i * temp.cols + j] = src->contents[i * src->cols + j];
    
    m_reduced_row_echelon(&temp);

    for(long i = 0; i < temp.rows; i++)
    {
        for(long j = 0; j < temp.rows; j++)
        {
            if(i == j && !is_equal(temp.contents[2 * i * temp.cols + j], 1.0))
            {
                m_destroy(&temp);
                return out;
            }
            else if(i != j && !is_equal(temp.contents[2 * i * temp.cols + j], 0.0))
            {
                m_destroy(&temp);
                return out;
            }
        }
    } 
    out = m_init(src->rows, src->cols);
    for(long i = 0; i < src->rows; i++)
        for(long j = 0; j < src->cols; j++)
            out.contents[i * out.cols + j] = temp.contents[i * temp.cols + temp.rows + j];
    
    m_destroy(&temp);
    return out;
}
/*Returns true iff all columns in matrix are linearly independent*/
int m_column_linear_independent(Matrix *src)
{
    Matrix temp;

    if(src->cols > src->rows)
        return 0;

    temp = m_copy(src);
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

    temp = m_transpose(src);
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

    temp = m_transpose(src);
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
/*Writes a matrix to a csv file. header should contain labels for each column like "x,y,z"*/
void m_export_csv(Matrix *src, char *filename, char *header)
{
    FILE *fp = fopen(filename, "w");

    fprintf(fp, "%s\n", header);
    for(long i = 0; i < src->rows; i++)
    {
        long j;
        for(j = 0; j < src->cols-1; j++)
        {
            fprintf(fp, "%lf,", src->contents[i * src->cols + j]);
        }
        if(j != 0)
            fprintf(fp, "%lf\n", src->contents[i * src->cols + j]);
    }
    fclose(fp);
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