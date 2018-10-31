// gaterecurrent2dnoind_cuda.c
#include <THC/THC.h>
#include <math.h>
#include "gaterecurrent2dnoind_cuda.h"
#include "cuda/gaterecurrent2dnoind_kernel.h"

// typedef bool boolean;

// this symbol will be resolved automatically from PyTorch libs
extern THCState *state;

int gaterecurrent2dnoind_forward_cuda(int horizontal_, int reverse_, THCudaTensor * X, THCudaTensor * G1, THCudaTensor * G2, THCudaTensor * G3, THCudaTensor * output)
{
	// Grab the input tensor to flat
	float * X_data = THCudaTensor_data(state, X);
	float * G1_data = THCudaTensor_data(state, G1);
	float * G2_data = THCudaTensor_data(state, G2);
	float * G3_data = THCudaTensor_data(state, G3);
	float * H_data = THCudaTensor_data(state, output);

	// dimensions
	int num_ = THCudaTensor_size(state, X, 0);
	int channels_ = THCudaTensor_size(state, X, 1);
	int height_ = THCudaTensor_size(state, X, 2);
	int width_ = THCudaTensor_size(state, X, 3);

	cudaStream_t stream = THCState_getCurrentStream(state);

	if(horizontal_ && !reverse_) // left to right
	{
		//const int count = height_ * channels_ * num_;
		Forward_left_right(num_, channels_, height_, width_, X_data, G1_data, G2_data, G3_data, H_data, horizontal_, reverse_, stream);
	}
	else if(horizontal_ && reverse_) // right to left
	{
		Forward_right_left(num_, channels_, height_, width_, X_data, G1_data, G2_data, G3_data, H_data, horizontal_, reverse_, stream);
	}
	else if(!horizontal_ && !reverse_) // top to bottom
	{
		Forward_top_bottom(num_, channels_, height_, width_, X_data, G1_data, G2_data, G3_data, H_data, horizontal_, reverse_, stream);
	}
	else
	{
		Forward_bottom_top(num_, channels_, height_, width_, X_data, G1_data, G2_data, G3_data, H_data, horizontal_, reverse_, stream);
	}

	return 1;
}

int gaterecurrent2dnoind_backward_cuda(int horizontal_, int reverse_, THCudaTensor* top, THCudaTensor* top_grad, THCudaTensor * X, THCudaTensor * G1, THCudaTensor * G2, THCudaTensor * G3, THCudaTensor * X_grad, THCudaTensor * G1_grad, THCudaTensor * G2_grad, THCudaTensor * G3_grad)
{
	//Grab the input tensor to flat
	float * X_data = THCudaTensor_data(state, X);
	float * G1_data = THCudaTensor_data(state, G1);
	float * G2_data = THCudaTensor_data(state, G2);
	float * G3_data = THCudaTensor_data(state, G3);
	float * H_data = THCudaTensor_data(state, top);

	float * H_diff = THCudaTensor_data(state, top_grad);

	float * X_diff = THCudaTensor_data(state, X_grad);
	float * G1_diff = THCudaTensor_data(state, G1_grad);
	float * G2_diff = THCudaTensor_data(state, G2_grad);
	float * G3_diff = THCudaTensor_data(state, G3_grad);

	// dimensions
	int num_ = THCudaTensor_size(state, X, 0);
	int channels_ = THCudaTensor_size(state, X, 1);
	int height_ = THCudaTensor_size(state, X, 2);
	int width_ = THCudaTensor_size(state, X, 3);

	cudaStream_t stream = THCState_getCurrentStream(state);

	if(horizontal_ && ! reverse_) //left to right
	{
		Backward_left_right(num_, channels_, height_, width_, X_data, G1_data, G2_data, G3_data, H_data, X_diff, G1_diff, G2_diff, G3_diff, H_diff, horizontal_, reverse_, stream);
	}
	else if(horizontal_ &&  reverse_) //right to left
	{
		Backward_right_left(num_, channels_, height_, width_, X_data, G1_data, G2_data, G3_data, H_data, X_diff, G1_diff, G2_diff, G3_diff, H_diff, horizontal_, reverse_, stream);
	}
	else if(!horizontal_ &&  !reverse_) //top to bottom
	{
		Backward_top_bottom(num_, channels_, height_, width_, X_data, G1_data, G2_data, G3_data, H_data, X_diff, G1_diff, G2_diff, G3_diff, H_diff, horizontal_, reverse_, stream);
	}
	else {
		Backward_bottom_top(num_, channels_, height_, width_, X_data, G1_data, G2_data, G3_data, H_data, X_diff, G1_diff, G2_diff, G3_diff, H_diff, horizontal_, reverse_, stream);
	}

	return 1;
}
