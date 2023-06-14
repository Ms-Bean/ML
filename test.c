#include "linalg/linalg_parallel.h"
int main(void)
{
    Matrix m = m_read_csv("data.csv");
    m_label_print(&m, "Data:");

    m_destroy(&m);

    return 0;
}