#include "mkl.h"
#include "ff.h"

#include <algorithm>

#include <cmath>

// batch_size, hidden_dim, layer_size
// IN is a batch_size,hidden_dim matrix;
// W1 is a layer_size,hidden_dim matrix;
// W2 is a layer_size,hidden_dim matrix;
// OUT is a batch_size,hidden_dim matrix; not necessarily assumed zeroed
void ff_l1(float* IN, float* W1, float* W2, float* OUT, int batch_size, int hidden_dim, int layer_size) {
	float* mi = IN;
	float* mo = OUT;
	const float sqrt2 = std::sqrt(2);

	for (int i = 0; i < batch_size; ++i) {
		float* mw1 = W1;
		float* mw2 = W2;
		for (int j = 0; j < layer_size; ++j) {
			float val = cblas_sdot_64(hidden_dim, mi, 1, mw1, 1);
			val = val * std::erfc(-val / sqrt2) / 2; // GELU activation
			cblas_saxpy_64(hidden_dim, val, mw2, 1, mo, 1);
			mw1 += hidden_dim;
			mw2 += hidden_dim;
		}
		mi += hidden_dim;
		mo += hidden_dim;
	}
}

// batch_size, hidden_dim, layer_size
// IN is a batch_size,hidden_dim matrix;
// W1 is a layer_size,hidden_dim matrix;
// W2 is a layer_size,hidden_dim matrix;
// OUT is a batch_size,hidden_dim matrix; assumed zeroed
void ff_l2(float* IN, float* W1, float* W2, float* OUT, int batch_size, int hidden_dim, int layer_size) {
	float* INTERMED = (float*)mkl_malloc(batch_size * layer_size * sizeof(float), 64);
	float* INTERMED2 = (float*)mkl_malloc(batch_size * layer_size * sizeof(float), 64);

	CBLAS_TRANSPOSE* transpose_instructions = (CBLAS_TRANSPOSE*)mkl_malloc(batch_size * sizeof(CBLAS_TRANSPOSE), 64);
	std::fill_n(transpose_instructions, batch_size, CBLAS_TRANSPOSE::CblasNoTrans);

	CBLAS_TRANSPOSE* transpose_instructions2 = (CBLAS_TRANSPOSE*)mkl_malloc(batch_size * sizeof(CBLAS_TRANSPOSE), 64);
	std::fill_n(transpose_instructions2, batch_size, CBLAS_TRANSPOSE::CblasTrans);

	MKL_INT* m_array = (MKL_INT*)mkl_malloc(batch_size * sizeof(MKL_INT), 64);
	std::fill_n(m_array, batch_size, layer_size);

	MKL_INT* n_array = (MKL_INT*)mkl_malloc(batch_size * sizeof(MKL_INT), 64);
	std::fill_n(n_array, batch_size, (MKL_INT)hidden_dim);

	MKL_INT* incs = (MKL_INT*)mkl_malloc(batch_size * sizeof(MKL_INT), 64);
	std::fill_n(incs, batch_size, (MKL_INT)1);

	float* alpha_array = (float*)mkl_malloc(batch_size * sizeof(float), 64);
	std::fill_n(alpha_array, batch_size, 1.f);

	float** W1_array = (float**)mkl_malloc(batch_size * sizeof(float*), 64);
	std::fill_n(W1_array, batch_size, W1);

	float** W2_array = (float**)mkl_malloc(batch_size * sizeof(float*), 64);
	std::fill_n(W2_array, batch_size, W2);

	float* beta_array = (float*)mkl_malloc(batch_size * sizeof(float), 64);
	std::fill_n(beta_array, batch_size, 0.f);

	float** in_pointers = (float**)mkl_malloc(batch_size * sizeof(float*), 64);
	for (int i = 0; i < batch_size; ++i) {
		in_pointers[i] = IN + hidden_dim;
	}

	float** intermed_pointers = (float**)mkl_malloc(batch_size * sizeof(float*), 64);
	for (int i = 0; i < batch_size; ++i) {
		intermed_pointers[i] = INTERMED + layer_size;
	}

	float** out_pointers = (float**)mkl_malloc(batch_size * sizeof(float*), 64);
	for (int i = 0; i < batch_size; ++i) {
		out_pointers[i] = OUT + hidden_dim;
	}

	MKL_INT* group_sizes = (MKL_INT*)mkl_malloc(batch_size * sizeof(MKL_INT), 64);
	std::fill_n(group_sizes, batch_size, 1);

	cblas_sgemv_batch_64(
		CBLAS_LAYOUT::CblasRowMajor,
		transpose_instructions,
		m_array,
		n_array,
		alpha_array,
		(const float**)W1_array,
		n_array,
		(const float**)in_pointers,
		incs,
		beta_array,
		intermed_pointers,
		incs,
		batch_size,
		group_sizes
	);

	// GELU activation
	vsCdfNorm(batch_size * layer_size, INTERMED, INTERMED2);
	vsMul(batch_size * layer_size, INTERMED, INTERMED2, INTERMED);

	cblas_sgemv_batch_64(
		CBLAS_LAYOUT::CblasRowMajor,
		transpose_instructions2,
		m_array,
		n_array,
		alpha_array,
		(const float**)W2_array,
		n_array,
		(const float**)intermed_pointers,
		incs,
		alpha_array, /* because we need per-batch-sample accumulation here */
		out_pointers,
		incs,
		batch_size,
		group_sizes
	);

	mkl_free(INTERMED);
	mkl_free(INTERMED2);
	mkl_free(transpose_instructions);
	mkl_free(transpose_instructions2);
	mkl_free(m_array);
	mkl_free(n_array);
	mkl_free(incs);
	mkl_free(alpha_array);
	mkl_free(W1_array);
	mkl_free(W2_array);
	mkl_free(beta_array);
	mkl_free(in_pointers);
	mkl_free(intermed_pointers);
	mkl_free(out_pointers);
}

// batch_size, hidden_dim, layer_size
// IN is a batch_size,hidden_dim matrix;
// W1 is a layer_size,hidden_dim matrix;
// W2 is a layer_size,hidden_dim matrix;
// OUT is a batch_size,hidden_dim matrix; not necessarily assumed zeroed
void ff_l3(float* IN, float* W1, float* W2, float* OUT, int batch_size, int hidden_dim, int layer_size) {
	float* INTERMED = (float*)mkl_malloc(batch_size * layer_size * sizeof(float), 64);
	float* INTERMED2 = (float*)mkl_malloc(batch_size * layer_size * sizeof(float), 64);

	// FOR THIS CALL ONLY:
	// op(IN) is an m-by-k matrix, so since op(IN)=NoTrans, m:=batch_size, k:=hidden_dim
	// op(W1) is a k-by-n matrix, so since op(W1)=Trans, k=hidden_dim (all good), n:=layer_size
	// INTERMED is a m-by-n matrix, m=batch_size (all good), thus - n must be layer_size
	cblas_sgemm_64(
		CBLAS_LAYOUT::CblasRowMajor,
		CBLAS_TRANSPOSE::CblasNoTrans,
		CBLAS_TRANSPOSE::CblasTrans,
		batch_size, layer_size, hidden_dim,
		1.f, IN, hidden_dim, W1, hidden_dim, 0.f, INTERMED, layer_size);


	// GELU activation
	vsCdfNorm(batch_size * layer_size, INTERMED, INTERMED2);
	vsMul(batch_size * layer_size, INTERMED, INTERMED2, INTERMED);

	// FOR THIS CALL ONLY:
	// op(INTERMED) is an m-by-k matrix, so since op(INTERMED)=NoTrans, m:=batch_size, k:=layer_size
	// op(W2) is a k-by-n matrix, so since op(W2)=NoTrans, k=layer_size (all good), n:=hidden_dim
	// OUT is a m-by-n matrix, so m=batch_size, n=hidden_dim
	cblas_sgemm_64(
		CBLAS_LAYOUT::CblasRowMajor,
		CBLAS_TRANSPOSE::CblasNoTrans,
		CBLAS_TRANSPOSE::CblasNoTrans,
		batch_size, hidden_dim, layer_size,
		1.f, INTERMED, layer_size, W2, hidden_dim, 0.f, OUT, hidden_dim);

	mkl_free(INTERMED);
	mkl_free(INTERMED2);
}
