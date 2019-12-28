// gaterecurrent2dnoind_cuda.c
//#include <THC/THC.h>
#include <math.h>
#include <torch/extension.h>
#include "gaterecurrent2dnoind_kernel.h"

int gaterecurrent2dnoind_forward_cuda(int horizontal_, int reverse_, torch::Tensor X, torch::Tensor G1, torch::Tensor G2, torch::Tensor G3, torch::Tensor output)
{
	// Grab the input tensor to flat
	float * X_data = X.data<float>();
	float * G1_data = G1.data<float>();
	float * G2_data = G2.data<float>();
	float * G3_data = G3.data<float>();
	float * H_data = output.data<float>();

	// dimensions
	int num_ =  X.size(0);
	int channels_ = X.size(1);
	int height_ = X.size(2);
	int width_ = X.size(3);


	if(horizontal_ && !reverse_) // left to right
	{
		//const int count = height_ * channels_ * num_;
		Forward_left_right(num_, channels_, height_, width_, X_data, G1_data, G2_data, G3_data, H_data, horizontal_, reverse_);
	}
	else if(horizontal_ && reverse_) // right to left
	{
		Forward_right_left(num_, channels_, height_, width_, X_data, G1_data, G2_data, G3_data, H_data, horizontal_, reverse_);
	}
	else if(!horizontal_ && !reverse_) // top to bottom
	{
		Forward_top_bottom(num_, channels_, height_, width_, X_data, G1_data, G2_data, G3_data, H_data, horizontal_, reverse_);
	}
	else
	{
		Forward_bottom_top(num_, channels_, height_, width_, X_data, G1_data, G2_data, G3_data, H_data, horizontal_, reverse_);
	}

	return 1;
}

int gaterecurrent2dnoind_backward_cuda(int horizontal_, int reverse_, torch::Tensor top, torch::Tensor top_grad, torch::Tensor X, torch::Tensor G1, torch::Tensor G2, torch::Tensor G3, torch::Tensor X_grad, torch::Tensor G1_grad, torch::Tensor G2_grad, torch::Tensor G3_grad)
{
	//Grab the input tensor to flat
	float * X_data = X.data<float>();
	float * G1_data = G1.data<float>();
	float * G2_data = G2.data<float>();
	float * G3_data = G3.data<float>();
	float * H_data = top.data<float>();

	float * H_diff = top_grad.data<float>();

	float * X_diff = X_grad.data<float>();
	float * G1_diff = G1_grad.data<float>();
	float * G2_diff = G2_grad.data<float>();
	float * G3_diff = G3_grad.data<float>();

	// dimensions
	int num_ = X.size(0);
	int channels_ = X.size(1);
	int height_ = X.size(2);
	int width_ = X.size(3);


	if(horizontal_ && ! reverse_) //left to right
	{
		Backward_left_right(num_, channels_, height_, width_, X_data, G1_data, G2_data, G3_data, H_data, X_diff, G1_diff, G2_diff, G3_diff, H_diff, horizontal_, reverse_);
	}
	else if(horizontal_ &&  reverse_) //right to left
	{
		Backward_right_left(num_, channels_, height_, width_, X_data, G1_data, G2_data, G3_data, H_data, X_diff, G1_diff, G2_diff, G3_diff, H_diff, horizontal_, reverse_);
	}
	else if(!horizontal_ &&  !reverse_) //top to bottom
	{
		Backward_top_bottom(num_, channels_, height_, width_, X_data, G1_data, G2_data, G3_data, H_data, X_diff, G1_diff, G2_diff, G3_diff, H_diff, horizontal_, reverse_);
	}
	else {
		Backward_bottom_top(num_, channels_, height_, width_, X_data, G1_data, G2_data, G3_data, H_data, X_diff, G1_diff, G2_diff, G3_diff, H_diff, horizontal_, reverse_);
	}

	return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
) {
m.def("forward", &gaterecurrent2dnoind_forward_cuda, "InnerProduct forward (CUDA)");
m.def("backward", &gaterecurrent2dnoind_backward_cuda, "InnerProduct backward (CUDA)");
}