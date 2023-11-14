#pragma once

// k: batch size, m: hidden_dim, n: number of nodes, must be >= (2^depth - 1)
// IN is a k,m matrix;
// W1 is an n,m matrix;
// W2 is an n,m matrix;
// OUT is an k,m matrix; assumed zeroed at the beginning
void fff_l1(float* IN, float* W1, float* W2, float* OUT, int k, int m, int n, int depth);

// batch_size, hidden_dim, n_nodes
// IN is a batch_size,hidden_dim matrix;
// W1 is a n_nodes,hidden_dim matrix;
// W2 is a n_nodes,hidden_dim matrix;
// OUT is a batch_size,hidden_dim matrix; assumed zeroed at the beginning
void fff_l2(float* IN, float* W1, float* W2, float* OUT, int batch_size, int hidden_dim, int n_nodes, int depth);