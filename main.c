#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef struct {
    int rank;
    int * shape;
    float * data;
} tensor;

// create a new tensor
tensor* create_tensor(int rank, int * shape)
{

    tensor* t = (tensor*) malloc(sizeof(tensor));

    t->rank = rank;
    t->shape = (int*) malloc(rank * sizeof(int));

    for (int i = 0; i < rank; i++) {
        t->shape[i] = shape[i];
    }

    int num_elems = 1;
    for (int i = 0; i < rank; i++) {
        num_elems *= shape[i];
    }
    t->data = (float*) malloc(num_elems * sizeof(float));
    return t;
}

int tensor_num_elems(tensor * t)
{   
    // calculate the number of elements that the tensor
    // contains
    int num_elements = 1;
    for (int i = 0; i < t->rank; i++) {
        num_elements *= t->shape[i];
    }
    return num_elements;
}

// both of these methods return a slice of the given array
// from start_idx to end_idx inclusive
int* int_array_slice(int * array, int start_idx, int end_idx)
{
    int slice_size = end_idx - start_idx + 1;
    int * slice = (int *) malloc(slice_size * sizeof(int));
    for (int i = start_idx; i <= end_idx; i++) {
        slice[i - start_idx] = array[i];
    }
    return slice;
}
float* float_array_slice(float * array, int start_idx, int end_idx)
{
    int slice_size = end_idx - start_idx + 1;
    float * slice = (float *) malloc(slice_size * sizeof(float));
    for (int i = start_idx; i <= end_idx; i++) {
        slice[i - start_idx] = array[i];
    }
    return slice;
}

// this will return a slice of the tensor from the first
// dimension of the tensor, at the index d0
// for clarity, if t is a matrix, then this function will
// return the row at index d0
tensor* tensor_slice_dim0(tensor * t, int d0)
{
    // create the new tensor
    tensor * slice = (tensor*) malloc(sizeof(tensor));
    slice->rank = t->rank - 1;
    slice->shape = int_array_slice(t->shape, 1, t->rank);

    // create the data of the new tensor
    // also for clarity: num_elems is the number of elements of the 
    // slice, NOT of the source tensor
    int num_elems = 1;
    for (int i = 0; i < slice->rank; i++) {
        num_elems *= slice->shape[i];
    }
    slice->data = (float *) malloc(num_elems * sizeof(float));

    // find what chunk of the tensor's data to 
    // extract from
    int start_idx = d0 * num_elems;
    int end_idx   = (d0+1) * num_elems;

    // load the slice of data from the source tensor on to the 
    // slice tensor's data
    int j = 0;
    for (int i = start_idx; i < end_idx; i++) {
        slice->data[j] = t->data[i];
        j++;
    }
    return slice;
}

void free_tensor(tensor* t) 
{
    free(t->shape);
    free(t->data);
    free(t);
}

float get_tensor_element(tensor* t, int* indices) 
{
    int idx = 0;
    for (int i = 0; i < t->rank; i++) {
        idx = idx * t->shape[i] + indices[i];
    }
    return t->data[idx];
}

void set_tensor_element(tensor* t, int* indices, float val) 
{
    int idx = 0;
    for (int i = 0; i < t->rank; i++) {
        idx = idx * t->shape[i] + indices[i];
    }
    t->data[idx] = val;
}

// sets every element in a tensor to a specific value
void broadcast_set(tensor * t, float value)
{
    // calculate the number of elements that the tensor
    // contains
    int num_elements = 1;
    for (int i = 0; i < t->rank; i++) {
        num_elements *= t->shape[i];
    }

    // loop through the data of the tensor, and set each 
    // element to the value
    for (int i = 0; i < num_elements; i++) {
        t->data[i] = value;
    }
}

// add a certain value to every element of a tensor
void broadcast_add(tensor * t, float value) 
{
    // calculate the number of elements that the tensor
    // contains
    int num_elements = 1;
    for (int i = 0; i < t->rank; i++) {
        num_elements *= t->shape[i];
    }
    // loop through the data of the tensor and add the value to 
    // each of the elements
    for (int i = 0; i < num_elements; i++) {
        t->data[i] += value;
    }
}

// multiply each element of the tensor by a particular value
void broadcast_multiply(tensor * t, float value) 
{
    // calculate the number of elements that the tensor
    // contains
    int num_elements = 1;
    for (int i = 0; i < t->rank; i++) {
        num_elements *= t->shape[i];
    }
    // loop through the data of the tensor and multiply the value by
    // each of the elements
    for (int i = 0; i < num_elements; i++) {
        t->data[i] *= value;
    }
}

// every value of t becomes equal to exp(t)
void tensor_exp(tensor * t) 
{
    int num_elems = tensor_num_elems(t);
    for (int i = 0; i < num_elems; i++) {
        float data_i = t->data[i];
        data_i = exp(data_i);
        t->data[i] = data_i;
    }
}

// just calculate the sum of every element in the tensor
float tensor_sum_d0(tensor * t)
{
    int num_elems = tensor_num_elems(t);
    float summation = 0;
    for (int i = 0 ; i < num_elems; i++) {
        summation += t->data[i];
    }
    return summation;
}

// map a function on to every element of a tensor
void tensor_map_d0(tensor * t, float (*map_function)(float) )
{
    int num_elems = tensor_num_elems(t);
    for (int i = 0; i < num_elems; i++) {
        float map_val = map_function(t->data[i]);
        t->data[i] = map_val;
    }
}
/*
// calculate softmax on the tensor, over every single element in the tensor
void tensor_softmax_d0(tensor * t)
{
    // get the exponent
    tensor_exp(t);
    float sum = tensor_sum_d0(t);
}
*/
tensor* matmul(tensor* m1, tensor* m2)
{
    //check for shape compatibility
    if (m1->shape[m1->rank-1] != m2->shape[0]){
        fprintf(stderr, "Error: incompatible shapes for matrix multiplication\n");
        return NULL;
    }

    int result_shape[2] = {m1->shape[0], m2->shape[1]};
    tensor* m3 = create_tensor(2, result_shape);

    for (int i = 0; i < m1->shape[0]; i++) {
        for (int j = 0; j < m2->shape[1]; j++) {
            float dot_prod = 0.0;
            for (int k = 0; k < m1->shape[1]; k++) {
                int m1_idx[2] = {i, k};
                int m2_idx[2] = {k, j};
                dot_prod += get_tensor_element(m1, m1_idx) * get_tensor_element(m2, m2_idx);
            }
            int m3_idx[2] = {i,j};
            set_tensor_element(m3, m3_idx, dot_prod);
        }
    }
    return m3;
}



void print_tensor(tensor * t) {

    printf("\n");

    if (t->rank == 1) {
        printf("[");
        for (int i = 0; i < t->shape[0]; i++) {
            int idx[1] = {i};
            float elem = get_tensor_element(t, idx);
            printf(" %f ", elem);
        }
        printf("]\n");
    } else {

        int R = t->shape[0];
        int C = t->shape[1];
        for (int i = 0; i < R; i++) {
            printf("[ ");
            for (int j = 0; j < C; j++) {
                int idx[2] = {i, j};
                float elem = get_tensor_element(t, idx);
                printf(" %f ",elem);
            }
            printf("]\n");
        }
    }
    printf("shape: (");
    for (int i = 0; i < t->rank; i++) {
        printf(" %d", t->shape[i]);
        if (i != t->rank-1) {
            printf(",");
        } else {
            printf(" ");
        }
    }
    printf(")\n");
}

/*
// calculate the determinant of a matrix
// calling this function assumes that the input tensor is 
// 2 dimensional
float recursive_determinant(tensor * mat)
{
    // if it is a 2x2 matrix, then just calculate the determinant 
    // by the difference of the product of the diagonals
    if (mat->shape[0] == 2 && mat->shape[1] == 2) {
        int idx_00[2] = {0,0};
        int idx_11[2] = {1,1};
        int idx_01[2] = {0,1};
        int idx_10[2] = {1,0};
        float d1 = get_tensor_element(mat, idx_00) * get_tensor_element(mat, idx_11);
        float d2 = get_tensor_element(mat, idx_01) * get_tensor_element(mat, idx_10);
        return d1 - d2;
    } else {
        // get the top row of the tensor
        tensor * top_row = tensor_slice_dim0(mat, 0);

        printf("top row: \n");
        print_tensor(top_row);

        // the determinant
        float det = 0.0;

        // the changing sign that will indicate whether to
        // add or subtract each recursive matrix
        float sign = 1.0;

        // the number of columns that mat has
        int mat_n_cols = mat->shape[1];
        // the number of rows that mat has
        int mat_n_rows = mat->shape[0];

        // loop through the top row, and find the values of
        // the sub-determinants recursively
        for (int i = 0; i < mat_n_cols; i++) {
            // grab the matrix made by all the values that
            // are not in top row element i's row or column, 
            int non_overlap_shape[2] = {mat->shape[0] - 1, mat->shape[1] - 1};
            tensor * non_overlap = create_tensor(2,non_overlap_shape);

            print_tensor(non_overlap);

            int k = 0;
            int mat_elems = tensor_num_elems(mat);

            // ensure that j does not signify an element of mat that is
            // in it's first row, by starting at the element that 
            // is one column in to mat
            for (int j = mat_n_cols; j < mat_elems; j++) {
                // make sure that j is not in the column of i
                if (j % i != 0) {
                    non_overlap->data[k] = mat->data[j];
                    k++;
                }
            }

            print_tensor(non_overlap);

            // free the non_overlap matrix
            free_tensor(non_overlap);
            // flip the sign for each element
            sign = -sign;
        }

        //free the top row
        free_tensor(top_row);

        return det;
    }
}
*/

int main()
{
    int shape0[2] = {3,3};
    tensor* t0 = create_tensor(2, shape0);
    float k = 0;
    for (int i = 0; i < tensor_num_elems(t0); i++) {
        t0->data[i] = k;
        k++;
    }

    print_tensor(t0);

    tensor_exp(t0);

    print_tensor(t0);
   

    return 0;
}