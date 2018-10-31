
// #include <stdbool.h>
// gaterecurrent2dnoind_cuda.h
int gaterecurrent2dnoind_forward_cuda(int horizontal_, int reverse_, THCudaTensor * X, THCudaTensor * G1, THCudaTensor * G2, THCudaTensor * G3, THCudaTensor * output);

int gaterecurrent2dnoind_backward_cuda(int horizontal_, int reverse_, THCudaTensor* top, THCudaTensor* top_grad, THCudaTensor * X, THCudaTensor * G1, THCudaTensor * G2, THCudaTensor * G3, THCudaTensor * X_diff, THCudaTensor * G1_diff, THCudaTensor * G2_diff, THCudaTensor * G3_diff);
