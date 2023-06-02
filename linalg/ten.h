#include <stdlib.h>
#include <math.h>
#include <stdarg.h>
#include <string.h>

typedef struct Tensor
{
    long rank;
    long *dims;
    
    double *contents;
} Tensor;

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
double dot_product(double *a, double *b, int len)
{
    double sum;
    sum = 0;
    for(long i = 0; i < len; i++)
        sum += a[i] * b[i];
    return sum;
}
void t_mult(Tensor *a, Tensor *b, Tensor *dst)
{
    Tensor b_tsp;
    long *reverse_indices;

    b_tsp = t_create_transpose(b);
}
void destroy(Tensor *t1)
{
    free(t1->contents);
    free(t1->dims);
}