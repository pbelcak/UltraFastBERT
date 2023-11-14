#include "fff.h"

#include "mkl.h"

#include <algorithm>
#include <cmath>

// k: batch size, m: hidden_dim, n: number of nodes, must be >= (2^depth - 1)
// IN is a k,m matrix;
// W1 is an n,m matrix;
// W2 is an n,m matrix;
// OUT is an k,m matrix; assumed zeroed at the beginning
void fff_l1(float* IN, float* W1, float* W2, float* OUT, int k, int m, int n, int depth) {
	size_t* current_nodes = (size_t*)mkl_calloc(k, sizeof(size_t), 64);
	const float sqrt2 = std::sqrt(2);
	
	for (int d = 0; d < depth;++d) {
		float* mi = IN;
		float* mo = OUT;
		for (int i = 0; i < k; ++i) {
			float val = cblas_sdot_64(m, mi, 1, W1+(current_nodes[i] * m), 1);
			val = val * std::erfc(-val / sqrt2) / 2; // GELU activation
			cblas_saxpy_64(m, val, OUT + (current_nodes[i] * m), 1, mo, 1);
		
			current_nodes[i] = 2 * current_nodes[i] + 1 + (val > 0.f ? 1 : 0);

			mi += m;
			mo += m;
		}
	}

	mkl_free(current_nodes);
}

// batch_size, hidden_dim, n_nodes
// IN is a batch_size,hidden_dim matrix;
// W1 is a n_nodes,hidden_dim matrix;
// W2 is a n_nodes,hidden_dim matrix;
// OUT is a batch_size,hidden_dim matrix; assumed zeroed at the beginning
void fff_l2(float* IN, float* W1, float* W2, float* OUT, int batch_size, int hidden_dim, int n_nodes, int depth) {
	size_t* current_nodes = (size_t*)mkl_calloc(batch_size, sizeof(size_t), 64);
	float* intermed = (float*)mkl_malloc(batch_size * sizeof(float), 64);
	float* intermed2 = (float*)mkl_malloc(batch_size * sizeof(float), 64);

	CBLAS_TRANSPOSE* transpose_instructions = (CBLAS_TRANSPOSE*)mkl_malloc(batch_size * sizeof(CBLAS_TRANSPOSE), 64);
	std::fill_n(transpose_instructions, batch_size, CBLAS_TRANSPOSE::CblasNoTrans);

	MKL_INT* m_array = (MKL_INT*)mkl_malloc(batch_size * sizeof(MKL_INT), 64);
	std::fill_n(m_array, batch_size, 1);

	MKL_INT* n_array = (MKL_INT*)mkl_malloc(batch_size * sizeof(MKL_INT), 64);
	std::fill_n(n_array, batch_size, (MKL_INT)hidden_dim);

	MKL_INT* incs = (MKL_INT*)mkl_malloc(batch_size * sizeof(MKL_INT), 64);
	std::fill_n(incs, batch_size, (MKL_INT)1);

	float* alpha_array = (float*)mkl_malloc(batch_size * sizeof(float), 64);
	std::fill_n(alpha_array, batch_size, 1.f);

	float* beta_array = (float*)mkl_malloc(batch_size * sizeof(float), 64);
	std::fill_n(beta_array, batch_size, 0.f);

	float** w1_pointers = (float**)mkl_malloc(batch_size * sizeof(float*), 64);
	std::fill_n(w1_pointers, batch_size, W1);

	float** w2_pointers = (float**)mkl_malloc(batch_size * sizeof(float*), 64);
	// std::fill_n(w1_pointers, batch_size, W2); will be done implicitly

	float** in_pointers = (float**)mkl_malloc(batch_size * sizeof(float*), 64);
	for (int i = 0; i < batch_size; ++i) {
		in_pointers[i] = IN + hidden_dim;
	}

	float** out_pointers = (float**)mkl_malloc(batch_size * sizeof(float*), 64);
	for (int i = 0; i < batch_size; ++i) {
		out_pointers[i] = OUT + hidden_dim;
	}

	float** intermed_pointers = (float**)mkl_malloc(batch_size * sizeof(float*), 64);
	for (int i = 0; i < batch_size; ++i) {
		intermed_pointers[i] = intermed + i;
	}

	for (int d = 0; d < depth; ++d) {
		cblas_sgemv_batch_64(
			CBLAS_LAYOUT::CblasRowMajor,
			transpose_instructions,
			m_array,
			n_array,
			alpha_array,
			(const float**)in_pointers,
			n_array,
			(const float**)w1_pointers,
			incs,
			beta_array,
			intermed_pointers,
			incs,
			batch_size,
			m_array /* also happens to be a (batch_size,) tensor of long 1s */
		);

		for (int k = 0; k < batch_size; ++k) {
			w2_pointers[k] = W2 + hidden_dim * current_nodes[k];
			current_nodes[k] = 2 * current_nodes[k] + 1 + (intermed[k] > 0.f ? 1 : 0);
			w1_pointers[k] = W1 + hidden_dim * current_nodes[k];
		}

		// gelu activation across the whole batch
		vsCdfNorm(batch_size, intermed, intermed2);
		vsMul(batch_size, intermed, intermed2, intermed);

		cblas_saxpy_batch_64(
			n_array,
			alpha_array,
			(const float**)w2_pointers,
			incs,
			out_pointers,
			incs,
			batch_size,
			m_array /* also happens to be a (batch_size,) tensor of long 1s */
		);
	}

	mkl_free(current_nodes);
	mkl_free(intermed);
	mkl_free(intermed2);
	mkl_free(transpose_instructions);
	mkl_free(m_array);
	mkl_free(n_array);
	mkl_free(incs);
	mkl_free(alpha_array);
	mkl_free(beta_array);
	mkl_free(w1_pointers);
	mkl_free(w2_pointers);
	mkl_free(in_pointers);
	mkl_free(intermed_pointers);
}