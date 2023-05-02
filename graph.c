#include <stdlib.h>
#include <stdio.h>

typedef struct scalar {
    float value;
    float grad;
    struct node* node_ptr;
} scalar;

typedef struct node {
    int num_scalars;
    struct scalar** scalars;
    struct node_function* function;
} node;

typedef struct node_function {
    scalar* (*aggregate_function)(scalar**, int);
} node_function;


//create a new node
node* create_node(int num_scalars, node_function* function)
{
    node* n = (node*) malloc(sizeof(node));

    n->num_scalars = num_scalars;
    //n->scalars = (scalar**) malloc(num_scalars * sizeof(scalar*));
    n->scalars = malloc(sizeof(scalar*) * num_scalars);
    for (int i = 0; i < num_scalars; i++) {
        n->scalars[i] = NULL;
    }
    n->function = function;
    return n;
}

//create a new scalar
scalar* create_scalar(float value, node* node_ptr)
{
    scalar * s = (scalar*) malloc(sizeof(scalar));
    s->value = value;
    s->grad = 0.0;
    s->node_ptr = node_ptr;
    return s;
}

//create a new node function
node_function* create_node_function(scalar* (*aggregate_function)(scalar**, int))
{
    node_function* nf = malloc(sizeof(node_function));
    nf->aggregate_function = aggregate_function;
    return nf;
}

void free_node(node* n) 
{
    free(n->scalars);
    free(n);
}

void free_scalar(scalar * s) 
{
    free(s);
}


// set the node's particular scalar to a new scalar 
void set_scalar_node(node * n, int index, scalar* new_scalar)
{
    if (index >= n->num_scalars || index < 0) 
    {
        // index out of bounds error
        return;
    }

    scalar* old_scalar = n->scalars[index];
    if (old_scalar != NULL) {
        // free the memory used by the old scalar
        free_scalar(old_scalar);
    }
    n->scalars[index] = new_scalar;
}

scalar* aggregate_node_scalars(node* n) {
    return n->function->aggregate_function(n->scalars, n->num_scalars);
}

// this is a simple function that will take
// the sum of all it's input scalars
scalar* summation(scalar** in, int n_in)
{
    float sum = 0;
    for (int i = 0; i < n_in; i++) {
        sum += in[i]->value;
    }
    scalar* result = create_scalar(sum, NULL);
    return result;
}

void trace_grad(scalar * _scalar)
{   
    // grab the scalar's node
    node * n = _scalar->node_ptr;
    int num_childeren = n->num_scalars;
    
    for (int i =0; i < num_childeren; i++) {

    }
}


int main()
{
    node_function* sum = create_node_function(summation);

    node * n0 = create_node(2, sum);
    scalar * s1 = create_scalar(1.0, NULL);
    scalar * s2 = create_scalar(2.0, NULL);

    n0->scalars[0] = s1;
    n0->scalars[1] = s2;
    
    scalar* result = aggregate_node_scalars(n0);

    printf("the result of the output graph is: %f\n", result->value);

    free_node(n0);
    free_scalar(s1);
    free_scalar(s2);
    free_scalar(result);

    return 0;
}