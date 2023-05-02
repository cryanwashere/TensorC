#include <stdlib.h>
#include <stdio.h>

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
        num_elems += shape[i];
    }
    t->data = (float*) malloc(num_elems * sizeof(float));
    return t;
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
    for (int i = start_idx; i < end_idx; i++) {
        
    }
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


// assuming that the rank of the tensor is 2
void print_tensor(tensor * t) {

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
        return d1 * d2;
    }

    
}


int main()
{
    int shape0[2] = {2,4};
    tensor* t0 = create_tensor(2,shape0);

    int shape1[2] = {4,2};
    tensor* t1 = create_tensor(2, shape1);

    broadcast_set(t0, 1.0);
    broadcast_set(t1, 1.0);

    tensor* t3 = matmul(t0, t1);

    broadcast_add(t3, 0.5);

    print_tensor(t3);

    //float det = recursive_determinant(t3);
    //printf("matrix determinant: %f", det);


    


    free_tensor(t0);

    return 0;
}