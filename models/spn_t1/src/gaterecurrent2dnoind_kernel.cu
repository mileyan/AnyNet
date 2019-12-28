#include <stdio.h>
#include <math.h>
#include <float.h>
#include "gaterecurrent2dnoind_kernel.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

__device__ void get_gate_idx_sf(int h1, int w1, int h2, int w2, int * out, int horizontal, int reverse)
{
	if(horizontal && ! reverse) // left -> right
	{
		if(w1>w2)
		{
			out[0]=h1;
			out[1]=w1;
		}
		else
		{
			out[0]=h2;
			out[1]=w2;
		}
	}
	if(horizontal && reverse)  // right -> left
	{
		if(w1<w2)
		{
			out[0]=h1;
			out[1]=w1;
		}
		else
		{
			out[0]=h2;
			out[1]=w2;
		}
	}
	if(!horizontal && !reverse)  // top  -> bottom
	{
		if(h1>h2)
		{
			out[0]=h1;
			out[1]=w1;
		}
		else
		{
			out[0]=h2;
			out[1]=w2;
		}
	}
	if(!horizontal && reverse)  // bottom -> top
	{
		if(h1<h2)
		{
			out[0]=h1;
			out[1]=w1;
		}
		else
		{
			out[0]=h2;
			out[1]=w2;
		}
	}

}

__device__ float get_data_sf(float * data, int num, int channels,int height, int width,int n,int c,int h,int w)
{
	if(h<0 || h >=height)
		return 0;
	if(w<0 || w >= width)
		return 0;

	return data[n*channels*height*width + c * height*width + h * width + w];
}

__device__ void set_data_sf(float * data, int num, int channels,int height, int width,int n,int c,int h,int w, float v)
{
	if(h<0 || h >=height)
		return ;
	if(w<0 || w >= width)
		return ;

	data[n*channels*height*width + c * height*width + h * width + w]=v;
}

__device__ float get_gate_sf(float * data, int num, int channels,int height, int width,int n,int c,int h1,int w1,int h2,int w2,int horizontal,int reverse)
{
	if(h1<0 || h1 >=height)
		return 0;
	if(w1<0 || w1 >= width)
		return 0;
	if(h2<0 || h2 >=height)
		return 0;
	if(w2<0 || w2 >= width)
		return 0;
	int idx[2];

	get_gate_idx_sf(h1,w1,h2,w2, idx,horizontal, reverse);

	int h = idx[0];
	int w = idx[1];

	return data[n*channels*height*width + c * height*width + h * width + w];
}

__device__ void set_gate_sf(float * data, int num, int channels,int height, int width,int n,int c,int h1,int w1,int h2,int w2,int horizontal,int reverse, float v)
{
	if(h1<0 || h1 >=height)
		return ;
	if(w1<0 || w1 >= width)
		return ;
	if(h2<0 || h2 >=height)
		return ;
	if(w2<0 || w2 >= width)
		return ;
	int idx[2];

	get_gate_idx_sf(h1,w1,h2,w2, idx,horizontal, reverse);

	int h = idx[0];
	int w = idx[1];

	data[n*channels*height*width + c * height*width + h * width + w]=v;
}

// we do not use set_gate_add_sf(...) in the caffe implimentation
// avoid using atomicAdd

__global__ void forward_one_col_left_right( int count, int T, int num,int channels, int height,  int width, float* X,  float* G1,  float* G2, float* G3, float* H, int horizontal, int reverse) {
  CUDA_1D_KERNEL_LOOP(index, count) {

  	int hc_count = height * channels;

  	int n,c,h,w;
  	int temp=index;
  	w = T;
  	n = temp / hc_count;
  	temp = temp % hc_count;
  	c = temp / height;
  	temp = temp % height;
  	h = temp;


  	float x_data = get_data_sf(X,num,channels,height,width,n,c,h,w);

  	float g_data_1 = get_gate_sf(G1,num,channels,height,width,n,c,h,w,h-1,w-1,horizontal,reverse);
  	float h_minus1_data_1 = get_data_sf(H,num,channels,height,width,n,c,h-1,w-1);
  	float h1_minus1 = g_data_1 * h_minus1_data_1;

  	float g_data_2 = get_gate_sf(G2,num,channels,height,width,n,c,h,w,h,w-1,horizontal,reverse);
  	float h_minus1_data_2 = get_data_sf(H,num,channels,height,width,n,c,h,w-1);
  	float h2_minus1 = g_data_2 * h_minus1_data_2;

  	float g_data_3 = get_gate_sf(G3,num,channels,height,width,n,c,h,w,h+1,w-1,horizontal,reverse);
  	float h_minus1_data_3 = get_data_sf(H,num,channels,height,width,n,c,h+1,w-1);
  	float h3_minus1 = g_data_3 * h_minus1_data_3;

  	float h_hype = h1_minus1 + h2_minus1 + h3_minus1;
  	float x_hype = (1 - g_data_1 - g_data_2 - g_data_3) * x_data;

  	float h_data = x_hype + h_hype;

  	set_data_sf(H,num,channels,height,width,n,c,h,w,h_data);

  }
}

__global__ void forward_one_col_right_left( int count, int T, int num,int channels, int height,  int width, float* X,  float* G1,  float* G2, float* G3, float* H,int horizontal,int reverse) {
  CUDA_1D_KERNEL_LOOP(index, count) {

  	int hc_count = height * channels;
  	int n,c,h,w;
  	int temp=index;
  	w = T;
  	n = temp / hc_count;
  	temp = temp % hc_count;
  	c = temp / height;
  	temp = temp % height;
  	h = temp;

  	float x_data = get_data_sf(X,num,channels,height,width,n,c,h,w);

  	float g_data_1 = get_gate_sf(G1,num,channels,height,width,n,c,h,w,h-1,w+1,horizontal,reverse);
  	float h_minus1_data_1 = get_data_sf(H,num,channels,height,width,n,c,h-1,w+1);
  	float h1_minus1 = g_data_1 * h_minus1_data_1;

  	float g_data_2 = get_gate_sf(G2,num,channels,height,width,n,c,h,w,h,w+1,horizontal,reverse);
  	float h_minus1_data_2 = get_data_sf(H,num,channels,height,width,n,c,h,w+1);
  	float h2_minus1 = g_data_2 * h_minus1_data_2;

  	float g_data_3 = get_gate_sf(G3,num,channels,height,width,n,c,h,w,h+1,w+1,horizontal,reverse);
  	float h_minus1_data_3 = get_data_sf(H,num,channels,height,width,n,c,h+1,w+1);
  	float h3_minus1 = g_data_3 * h_minus1_data_3;

  	float h_hype = h1_minus1 + h2_minus1 + h3_minus1;
  	float x_hype = (1 - g_data_1 - g_data_2 - g_data_3) * x_data;

  	float h_data = x_hype + h_hype;

  	set_data_sf(H,num,channels,height,width,n,c,h,w,h_data);

  }
}

__global__ void forward_one_row_top_bottom( int count, int T, int num,int channels, int height,  int width, float* X,  float* G1,  float* G2, float* G3, float* H,int horizontal,int reverse) {
  CUDA_1D_KERNEL_LOOP(index, count) {

  	int wc_count = width * channels;

  	int n,c,h,w;
  	int temp=index;
  	h = T;
  	n = temp / wc_count;
  	temp = temp % wc_count;
  	c = temp / width;
  	temp = temp % width;
  	w = temp;


  	float x_data = get_data_sf(X,num,channels,height,width,n,c,h,w);


  	float g_data_1 = get_gate_sf(G1,num,channels,height,width,n,c,h,w,h-1,w-1,horizontal,reverse);
  	float h_minus1_data_1 = get_data_sf(H,num,channels,height,width,n,c,h-1,w-1);
  	float h1_minus1 = g_data_1 * h_minus1_data_1;

  	float g_data_2 = get_gate_sf(G2,num,channels,height,width,n,c,h,w,h-1,w,horizontal,reverse);
  	float h_minus1_data_2 = get_data_sf(H,num,channels,height,width,n,c,h-1,w);
  	float h2_minus1 = g_data_2 * h_minus1_data_2;

  	float g_data_3 = get_gate_sf(G3,num,channels,height,width,n,c,h,w,h-1,w+1,horizontal,reverse);
  	float h_minus1_data_3 = get_data_sf(H,num,channels,height,width,n,c,h-1,w+1);
  	float h3_minus1 = g_data_3 * h_minus1_data_3;

  	float h_hype = h1_minus1 + h2_minus1 + h3_minus1;
  	float x_hype = (1 - g_data_1 - g_data_2 - g_data_3) * x_data;

  	float h_data = x_hype + h_hype;

  	set_data_sf(H,num,channels,height,width,n,c,h,w,h_data);

  }
}

__global__ void forward_one_row_bottom_top( int count, int T, int num,int channels, int height,  int width, float* X,  float* G1,  float* G2, float* G3, float* H,int horizontal,int reverse) {
  CUDA_1D_KERNEL_LOOP(index, count) {

  	int wc_count = width * channels;

  	int n,c,h,w;
  	int temp=index;
  	h = T;
  	n = temp / wc_count;
  	temp = temp % wc_count;
  	c = temp / width;
  	temp = temp % width;
  	w = temp;


  	float x_data = get_data_sf(X,num,channels,height,width,n,c,h,w);


  	float g_data_1 = get_gate_sf(G1,num,channels,height,width,n,c,h,w,h+1,w-1,horizontal,reverse);
  	float h_minus1_data_1 = get_data_sf(H,num,channels,height,width,n,c,h+1,w-1);
  	float h1_minus1 = g_data_1 * h_minus1_data_1;


  	float g_data_2 = get_gate_sf(G2,num,channels,height,width,n,c,h,w,h+1,w,horizontal,reverse);
  	float h_minus1_data_2 = get_data_sf(H,num,channels,height,width,n,c,h+1,w);
  	float h2_minus1 = g_data_2 * h_minus1_data_2;

  	float g_data_3 = get_gate_sf(G3,num,channels,height,width,n,c,h,w,h+1,w+1,horizontal,reverse);
  	float h_minus1_data_3 = get_data_sf(H,num,channels,height,width,n,c,h+1,w+1);
  	float h3_minus1 = g_data_3 * h_minus1_data_3;

  	float h_hype = h1_minus1 + h2_minus1 + h3_minus1;
  	float x_hype = (1 - g_data_1 - g_data_2 - g_data_3) * x_data;

  	float h_data = x_hype + h_hype;

  	set_data_sf(H,num,channels,height,width,n,c,h,w,h_data);

  }
}


__global__ void backward_one_col_left_right( int count, int T, int num,int channels, int height,  int width, float* X,  float* G1,  float* G2, float* G3,  float* H, float * X_diff, float * G1_diff,float* G2_diff,float * G3_diff, float * Hdiff,int horizontal,int reverse) {
  CUDA_1D_KERNEL_LOOP(index, count) {

  	int hc_count = height * channels;

  	int n,c,h,w;
  	int temp=index;

  	w = T;
  	n = temp / hc_count;
  	temp = temp % hc_count;
  	c = temp / height;
  	temp = temp % height;
  	h = temp;

  	float x_data = get_data_sf(X,num,channels,height,width,n,c,h,w);

  	//h(t)_diff = top(t)_diff
  	float h_diff = get_data_sf(Hdiff,num,channels,height,width,n,c,h,w);

  	//h(t)_diff += h(t+1)_diff * g(t+1) if t<T
  	float add1_h3_diff = get_data_sf(Hdiff,num,channels,height,width,n,c,h-1,w+1);
  	float add1_g3_data = get_gate_sf(G3,num,channels,height,width,n,c,h,w,h-1,w+1,horizontal,reverse);

  	float add1_h2_diff = get_data_sf(Hdiff,num,channels,height,width,n,c,h,w+1);
  	float add1_g2_data = get_gate_sf(G2,num,channels,height,width,n,c,h,w,h,w+1,horizontal,reverse);

  	float add1_h1_diff = get_data_sf(Hdiff,num,channels,height,width,n,c,h+1,w+1);
  	float add1_g1_data = get_gate_sf(G1,num,channels,height,width,n,c,h,w,h+1,w+1,horizontal,reverse);

  	h_diff = h_diff + add1_h3_diff * add1_g3_data + add1_h2_diff * add1_g2_data + add1_h1_diff * add1_g1_data;


  	//Hdiff[n*channels*height*width + c*height*width + h*width + w]=0;
  	set_data_sf(Hdiff,num,channels,height,width,n,c,h,w,h_diff);


  	//x(t)_diff=(1-sum(g_date))*h(t)_diff
    float g1_data =  get_gate_sf(G1,num,channels,height,width,n,c,h,w,h-1,w-1,horizontal,reverse);
  	float g2_data =  get_gate_sf(G2,num,channels,height,width,n,c,h,w,h,w-1,horizontal,reverse);
  	float g3_data =  get_gate_sf(G3,num,channels,height,width,n,c,h,w,h+1,w-1,horizontal,reverse);

  	float x_diff = (1- g1_data -g2_data -g3_data) * h_diff;
  	set_data_sf(X_diff,num,channels,height,width,n,c,h,w,x_diff);


  	// g_diff = h_diff * (h_data(t-1) - x_data)
  	float h1_minus1_data = get_data_sf(H,num,channels,height,width,n,c,h-1,w-1);
  	float g1_diff = h_diff * (h1_minus1_data - x_data);
  	set_gate_sf(G1_diff,num,channels,height,width,n,c,h,w,h-1,w-1,horizontal,reverse,g1_diff);

  	float h2_minus1_data = get_data_sf(H,num,channels,height,width,n,c,h,w-1);
  	float g2_diff = h_diff * (h2_minus1_data - x_data);
  	set_gate_sf(G2_diff,num,channels,height,width,n,c,h,w,h,w-1,horizontal,reverse,g2_diff);

  	float h3_minus1_data = get_data_sf(H,num,channels,height,width,n,c,h+1,w-1);
  	float g3_diff = h_diff * (h3_minus1_data - x_data);
  	set_gate_sf(G3_diff,num,channels,height,width,n,c,h,w,h+1,w-1,horizontal,reverse,g3_diff);

  }
}


__global__ void backward_one_col_right_left( int count, int T, int num,int channels, int height,  int width, float* X, float* G1,  float* G2, float* G3,  float* H, float * X_diff, float * G1_diff,float* G2_diff,float * G3_diff, float * Hdiff,int horizontal,int reverse) {
  CUDA_1D_KERNEL_LOOP(index, count) {

  	int hc_count = height * channels;

  	int n,c,h,w;
  	int temp=index;


  	w = T;
  	n = temp / hc_count;
  	temp = temp % hc_count;
  	c = temp / height;
  	temp = temp % height;
  	h = temp;


  	float x_data = get_data_sf(X,num,channels,height,width,n,c,h,w);

  	//h(t)_diff = top(t)_diff
  	float h_diff = get_data_sf(Hdiff,num,channels,height,width,n,c,h,w);

  	///h(t)_diff += h(t+1)_diff * g(t+1) if t<T
  	float add1_h3_diff = get_data_sf(Hdiff,num,channels,height,width,n,c,h-1,w-1);
  	float add1_g3_data = get_gate_sf(G3,num,channels,height,width,n,c,h,w,h-1,w-1,horizontal,reverse);

  	float add1_h2_diff = get_data_sf(Hdiff,num,channels,height,width,n,c,h,w-1);
  	float add1_g2_data = get_gate_sf(G2,num,channels,height,width,n,c,h,w,h,w-1,horizontal,reverse);

  	float add1_h1_diff = get_data_sf(Hdiff,num,channels,height,width,n,c,h+1,w-1);
  	float add1_g1_data = get_gate_sf(G1,num,channels,height,width,n,c,h,w,h+1,w-1,horizontal,reverse);

  	h_diff = h_diff + add1_h3_diff * add1_g3_data + add1_h2_diff * add1_g2_data + add1_h1_diff * add1_g1_data;


  	set_data_sf(Hdiff,num,channels,height,width,n,c,h,w,h_diff);

    float g1_data =  get_gate_sf(G1,num,channels,height,width,n,c,h,w,h-1,w+1,horizontal,reverse);
  	float g2_data =  get_gate_sf(G2,num,channels,height,width,n,c,h,w,h,w+1,horizontal,reverse);
  	float g3_data =  get_gate_sf(G3,num,channels,height,width,n,c,h,w,h+1,w+1,horizontal,reverse);
  	float x_diff = (1- g1_data -g2_data -g3_data) * h_diff;
  	set_data_sf(X_diff,num,channels,height,width,n,c,h,w,x_diff);

      // g_diff = h_diff * (h_data(t-1) - x_data)
  	float h1_minus1_data = get_data_sf(H,num,channels,height,width,n,c,h-1,w+1);
  	float g1_diff = h_diff * (h1_minus1_data - x_data);
  	set_gate_sf(G1_diff,num,channels,height,width,n,c,h,w,h-1,w+1,horizontal,reverse,g1_diff);


  	float h2_minus1_data = get_data_sf(H,num,channels,height,width,n,c,h,w+1);
  	float g2_diff = h_diff * (h2_minus1_data - x_data);
  	set_gate_sf(G2_diff,num,channels,height,width,n,c,h,w,h,w+1,horizontal,reverse,g2_diff);

  	float h3_minus1_data = get_data_sf(H,num,channels,height,width,n,c,h+1,w+1);
  	float g3_diff = h_diff * (h3_minus1_data - x_data);
  	set_gate_sf(G3_diff,num,channels,height,width,n,c,h,w,h+1,w+1,horizontal,reverse,g3_diff);


  }
}

__global__ void backward_one_row_top_bottom( int count, int T, int num,int channels, int height,  int width, float* X, float* G1,  float* G2, float* G3,  float* H, float * X_diff, float * G1_diff,float* G2_diff,float * G3_diff, float * Hdiff,int horizontal,int reverse) {
  CUDA_1D_KERNEL_LOOP(index, count) {


  	int wc_count = width * channels;

  	int n,c,h,w;
  	int temp=index;
  	h = T;
  	n = temp / wc_count;
  	temp = temp % wc_count;
  	c = temp / width;
  	temp = temp % width;
  	w = temp;

  	float x_data = get_data_sf(X,num,channels,height,width,n,c,h,w);

  	float h_diff = get_data_sf(Hdiff,num,channels,height,width,n,c,h,w);

  	//h(t)_diff += h(t+1)_diff * g(t+1) if t<T
  	float add1_h3_diff = get_data_sf(Hdiff,num,channels,height,width,n,c,h+1,w-1);
  	float add1_g3_data = get_gate_sf(G3,num,channels,height,width,n,c,h,w,h+1,w-1,horizontal,reverse);

  	float add1_h2_diff = get_data_sf(Hdiff,num,channels,height,width,n,c,h+1,w);
  	float add1_g2_data = get_gate_sf(G2,num,channels,height,width,n,c,h,w,h+1,w,horizontal,reverse);

  	float add1_h1_diff = get_data_sf(Hdiff,num,channels,height,width,n,c,h+1,w+1);
  	float add1_g1_data = get_gate_sf(G1,num,channels,height,width,n,c,h,w,h+1,w+1,horizontal,reverse);

  	h_diff = h_diff + add1_h3_diff * add1_g3_data + add1_h2_diff * add1_g2_data + add1_h1_diff * add1_g1_data;


  	set_data_sf(Hdiff,num,channels,height,width,n,c,h,w,h_diff);


  	//x(t)_diff=(1-g(t))*h(t)_diff
  	float g1_data =  get_gate_sf(G1,num,channels,height,width,n,c,h,w,h-1,w-1,horizontal,reverse);
  	float g2_data =  get_gate_sf(G2,num,channels,height,width,n,c,h,w,h-1,w,horizontal,reverse);
  	float g3_data =  get_gate_sf(G3,num,channels,height,width,n,c,h,w,h-1,w+1,horizontal,reverse);
  	float x_diff = (1- g1_data -g2_data -g3_data) * h_diff;
  	set_data_sf(X_diff,num,channels,height,width,n,c,h,w,x_diff);



  	// g_diff = h_diff * (h_data(t-1) - x_data)
  	float h1_minus1_data = get_data_sf(H,num,channels,height,width,n,c,h-1,w-1);
  	float g1_diff = h_diff * (h1_minus1_data - x_data);
  	set_gate_sf(G1_diff,num,channels,height,width,n,c,h,w,h-1,w-1,horizontal,reverse,g1_diff);

  	float h2_minus1_data = get_data_sf(H,num,channels,height,width,n,c,h-1,w);
  	float g2_diff = h_diff * (h2_minus1_data - x_data);
  	set_gate_sf(G2_diff,num,channels,height,width,n,c,h,w,h-1,w,horizontal,reverse,g2_diff);

  	float h3_minus1_data = get_data_sf(H,num,channels,height,width,n,c,h-1,w+1);
  	float g3_diff = h_diff * (h3_minus1_data - x_data);
  	set_gate_sf(G3_diff,num,channels,height,width,n,c,h,w,h-1,w+1,horizontal,reverse,g3_diff);

  }
}

__global__ void backward_one_row_bottom_top( int count, int T, int num,int channels, int height,  int width,  float* X,  float* G1,  float* G2, float* G3,  float* H, float * X_diff, float * G1_diff,float* G2_diff,float * G3_diff, float * Hdiff,int horizontal,int reverse) {
  CUDA_1D_KERNEL_LOOP(index, count) {

  	int wc_count = width * channels;

  	int n,c,h,w;
  	int temp=index;
  	h = T;
  	n = temp / wc_count;
  	temp = temp % wc_count;
  	c = temp / width;
  	temp = temp % width;
  	w = temp;

  	float x_data = get_data_sf(X,num,channels,height,width,n,c,h,w);

  	//h(t)_diff = top(t)_diff
  	float h_diff = get_data_sf(Hdiff,num,channels,height,width,n,c,h,w);

  	//h(t)_diff += h(t+1)_diff * g(t+1) if t<T
  	float add1_h3_diff = get_data_sf(Hdiff,num,channels,height,width,n,c,h-1,w-1);
  	float add1_g3_data = get_gate_sf(G3,num,channels,height,width,n,c,h,w,h-1,w-1,horizontal,reverse);

  	float add1_h2_diff = get_data_sf(Hdiff,num,channels,height,width,n,c,h-1,w);
  	float add1_g2_data = get_gate_sf(G2,num,channels,height,width,n,c,h,w,h-1,w,horizontal,reverse);

  	float add1_h1_diff = get_data_sf(Hdiff,num,channels,height,width,n,c,h-1,w+1);
  	float add1_g1_data = get_gate_sf(G1,num,channels,height,width,n,c,h,w,h-1,w+1,horizontal,reverse);

  	h_diff = h_diff + add1_h3_diff * add1_g3_data + add1_h2_diff * add1_g2_data + add1_h1_diff * add1_g1_data;


  	set_data_sf(Hdiff,num,channels,height,width,n,c,h,w,h_diff);


  	//x(t)_diff=(1-g(t))*h(t)_diff
  	float g1_data =  get_gate_sf(G1,num,channels,height,width,n,c,h,w,h+1,w-1,horizontal,reverse);
  	float g2_data =  get_gate_sf(G2,num,channels,height,width,n,c,h,w,h+1,w,horizontal,reverse);
  	float g3_data =  get_gate_sf(G3,num,channels,height,width,n,c,h,w,h+1,w+1,horizontal,reverse);
   	float x_diff = (1- g1_data -g2_data -g3_data) * h_diff;
  	set_data_sf(X_diff,num,channels,height,width,n,c,h,w,x_diff);


  	// g_diff = h_diff * (h_data(t-1) - x_data)
  	float h1_minus1_data = get_data_sf(H,num,channels,height,width,n,c,h+1,w-1);
  	float g1_diff = h_diff * (h1_minus1_data - x_data);
  	set_gate_sf(G1_diff,num,channels,height,width,n,c,h,w,h+1,w-1,horizontal,reverse,g1_diff);

  	//float g2_diff = h_diff * g2_idx * x_data * -1;
  	float h2_minus1_data = get_data_sf(H,num,channels,height,width,n,c,h+1,w);
  	float g2_diff = h_diff * (h2_minus1_data - x_data);
  	set_gate_sf(G2_diff,num,channels,height,width,n,c,h,w,h+1,w,horizontal,reverse,g2_diff);

  	//float g3_diff = h_diff * g3_idx * x_data * -1;
  	float h3_minus1_data = get_data_sf(H,num,channels,height,width,n,c,h+1,w+1);
  	float g3_diff = h_diff * (h3_minus1_data - x_data);
  	set_gate_sf(G3_diff,num,channels,height,width,n,c,h,w,h+1,w+1,horizontal,reverse,g3_diff);


  }
}


void Forward_left_right(int num_, int channels_, int height_, int width_,  float * X,  float * G1,  float * G2,  float * G3, float * H, int horizontal_, int reverse_)
{
   int count = height_ * channels_ * num_;
   int kThreadsPerBlock = 1024;
  cudaError_t err;

  for(int t=0; t<width_; t++) {
    	forward_one_col_left_right<<<(count + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(count, t, num_, channels_, height_, width_, X, G1, G2, G3, H, horizontal_, reverse_);

      err = cudaGetLastError();
      if(cudaSuccess != err)
      {
          fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
          exit( -1 );
      }
  }
  return ;
}

void Forward_right_left(int num_, int channels_, int height_, int width_,  float * X,  float * G1,  float * G2,  float * G3, float * H, int horizontal_, int reverse_)
{
   int count = height_ * channels_ * num_;
   int kThreadsPerBlock = 1024;
  cudaError_t err;

  for(int t = width_ - 1; t >= 0; t--) {
    	forward_one_col_right_left<<<(count + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(count, t, num_, channels_, height_, width_, X, G1, G2, G3, H, horizontal_, reverse_);

      err = cudaGetLastError();
      if(cudaSuccess != err)
      {
          fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
          exit( -1 );
      }
  }
  return ;
}

void Forward_top_bottom(int num_, int channels_, int height_, int width_,  float * X,  float * G1,  float * G2,  float * G3, float * H, int horizontal_, int reverse_)
{
   int count = width_ * channels_ * num_;
   int kThreadsPerBlock = 1024;
  cudaError_t err;

  for(int t=0; t< height_; t++) {
    	forward_one_row_top_bottom<<<(count + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(count, t, num_, channels_, height_, width_, X, G1, G2, G3, H, horizontal_, reverse_);

      err = cudaGetLastError();
      if(cudaSuccess != err)
      {
          fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
          exit( -1 );
      }
  }
  return ;
}

void Forward_bottom_top(int num_, int channels_, int height_, int width_,  float * X,  float * G1,  float * G2,  float * G3, float * H, int horizontal_, int reverse_)
{
   int count = width_ * channels_ * num_;
   int kThreadsPerBlock = 1024;
  cudaError_t err;

  for(int t = height_-1; t >= 0; t--) {
    	forward_one_row_bottom_top<<<(count + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(count, t, num_, channels_, height_, width_, X, G1, G2, G3, H, horizontal_, reverse_);

      err = cudaGetLastError();
      if(cudaSuccess != err)
      {
          fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
          exit( -1 );
      }
  }
  return ;
}

void Backward_left_right(int num_, int channels_, int height_, int width_,  float * X,  float * G1,  float * G2,  float * G3,  float * H, float * X_diff, float * G1_diff, float * G2_diff, float * G3_diff, float * H_diff, int horizontal_, int reverse_)
{
   int count =  height_ * channels_ * num_;
   int kThreadsPerBlock = 1024;
  cudaError_t err;

  for(int t = width_ -1; t>=0; t--)
  {
    backward_one_col_left_right<<<(count + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(count, t, num_, channels_, height_, width_, X, G1, G2, G3, H, X_diff, G1_diff, G2_diff, G3_diff, H_diff, horizontal_, reverse_);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }
  }
  return ;
}

void Backward_right_left(int num_, int channels_, int height_, int width_,  float * X,  float * G1,  float * G2,  float * G3,  float * H, float * X_diff, float * G1_diff, float * G2_diff, float * G3_diff, float * H_diff, int horizontal_, int reverse_)
{
   int count =  height_ * channels_ * num_;
   int kThreadsPerBlock = 1024;
  cudaError_t err;

  for(int t = 0; t<width_; t++)
  {
    backward_one_col_right_left<<<(count + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(count, t, num_, channels_, height_, width_, X, G1, G2, G3, H, X_diff, G1_diff, G2_diff, G3_diff, H_diff, horizontal_, reverse_);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }
  }
  return ;
}

void Backward_top_bottom(int num_, int channels_, int height_, int width_,  float * X,  float * G1,  float * G2,  float * G3,  float * H, float * X_diff, float * G1_diff, float * G2_diff, float * G3_diff, float * H_diff, int horizontal_, int reverse_)
{
   int count =  width_ * channels_ * num_;
   int kThreadsPerBlock = 1024;
  cudaError_t err;

  for(int t = height_-1; t>=0; t--)
  {
    backward_one_row_top_bottom<<<(count + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(count, t, num_, channels_, height_, width_, X, G1, G2, G3, H, X_diff, G1_diff, G2_diff, G3_diff, H_diff, horizontal_, reverse_);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }
  }
  return ;
}

void Backward_bottom_top(int num_, int channels_, int height_, int width_,  float * X,  float * G1,  float * G2,  float * G3,  float * H, float * X_diff, float * G1_diff, float * G2_diff, float * G3_diff, float * H_diff, int horizontal_, int reverse_)
{
   int count =  width_ * channels_ * num_;
   int kThreadsPerBlock = 1024;
  cudaError_t err;

  for(int t = 0; t<height_; t++)
  {
    backward_one_row_bottom_top<<<(count + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(count, t, num_, channels_, height_, width_, X, G1, G2, G3, H, X_diff, G1_diff, G2_diff, G3_diff, H_diff, horizontal_, reverse_);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }
  }
  return ;
}

