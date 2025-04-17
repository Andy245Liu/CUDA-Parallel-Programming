typedef struct CSR{
    float *data;
    int *col_index;
    int *row_ptr;
    int count;
    int rows;
    int cols;
}CSR;

typedef struct JDS{
    float* data;
    int* col_index;
    int* col_ptr;
    int* row_perm;
    int count;
    int max_nonzeros;
    int rows;
    int cols;
}JDS;


CSR* matrix_to_CSR(float* A, int M, int N);

JDS* CSR_to_transposed_JDS(CSR* csr, int M, int N);