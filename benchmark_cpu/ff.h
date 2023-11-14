#pragma once

// batch_size, hidden_dim, layer_size
// IN is a batch_size,hidden_dim matrix;
// W1 is a layer_size,hidden_dim matrix;
// W2 is a layer_size,hidden_dim matrix;
// OUT is a batch_size,hidden_dim matrix; assumed zeroed at the beginning
void ff_l1(float* IN, float* W1, float* W2, float* OUT, int batch_size, int hidden_dim, int layer_size);
void ff_l2(float* IN, float* W1, float* W2, float* OUT, int batch_size, int hidden_dim, int layer_size);
void ff_l3(float* IN, float* W1, float* W2, float* OUT, int batch_size, int hidden_dim, int layer_size);
