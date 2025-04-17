#include "sparse_matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <queue>
#include <utility>
#include <vector>

CSR* matrix_to_CSR(float* A, int M, int N){
    //M is number of rows ini matrix, N is number of columns in matrix
    float* temp_data = (float *) malloc(M*N*sizeof(float));
    int* temp_col_num = (int *) malloc(M * N * sizeof(int));
    int* row_index = (int*) malloc((M+1)*sizeof(int));
    int count = 0;
    for(int i = 0; i < M; ++i){
        row_index[i] = count;
        for(int j = 0; j < N; ++j){
            if(A[i*N + j] != 0){
                temp_data[count] = A[i*N + j];
                temp_col_num[count] = j;
                count++;

            }
        }
    }
    row_index[M] = count;
    CSR* myCSR = (CSR*)malloc(sizeof(CSR));
    myCSR->count = count;
    myCSR->row_ptr = row_index;
    myCSR->data = (float*) malloc(count*sizeof(float));
    std::copy(temp_data, temp_data+count, myCSR->data);
    myCSR->col_index = (int*) malloc(count*sizeof(int));
    std::copy(temp_col_num, temp_col_num+count, myCSR->col_index);
    
    myCSR->rows = M;
    myCSR->cols = N;

    free(temp_col_num);
    free(temp_data);

    return myCSR;
}


JDS* CSR_to_transposed_JDS(CSR* csr, int M, int N){
    //M is number of rows in the original matrix
    JDS* myJDS = (JDS*) malloc(sizeof(JDS));
    myJDS->count = csr->count; //jds count val filled here
    myJDS->rows = M; //jds rows val filled here
    myJDS->cols = N; //jds cols val filled here
    myJDS->data = (float*) malloc(myJDS->count * sizeof(float));
    myJDS->col_index = (int*) malloc(myJDS->count * sizeof(int));
    
    myJDS->row_perm = (int*) malloc(M * sizeof(int));
    std::priority_queue<std::pair<int,int>> pq;
    for(int i = 0; i < M; ++i){
        pq.push(std::make_pair(csr->row_ptr[i+1] - csr->row_ptr[i], i));
    }
    int max_num_of_nonzeros_in_a_row = pq.top().first;
    myJDS->max_nonzeros = max_num_of_nonzeros_in_a_row; //jds max_nonzeros val filled here
    myJDS->col_ptr = (int*) malloc((max_num_of_nonzeros_in_a_row+1) * sizeof(int));


    //fill in the data structure now
    int index = 0;
    while(!pq.empty()){
        auto[a,b] = pq.top();
        myJDS->row_perm[index] = b; //jds row_perm filled out here
        index++;
        pq.pop();
    }

    int curr_col_filling = 0;
    int tot_data_filled = 0;

    while(curr_col_filling < max_num_of_nonzeros_in_a_row){
        myJDS->col_ptr[curr_col_filling] = tot_data_filled;
        for(int i = 0; i < M; ++i){
            int row_we_are_filling = myJDS->row_perm[i];
            int csr_index = csr->row_ptr[row_we_are_filling]+curr_col_filling;
            if(csr_index < csr->row_ptr[row_we_are_filling+1]){
                myJDS->data[tot_data_filled] = csr->data[csr_index]; //jds data filled here
                myJDS->col_index[tot_data_filled] = csr->col_index[csr_index]; //jds col_index filled here
                tot_data_filled++;
            }

        }
        
        curr_col_filling++;

    }

    myJDS->col_ptr[curr_col_filling] = tot_data_filled; //to add the ptr to the last element
    return myJDS;
}

