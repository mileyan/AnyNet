#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

cd src/cuda/
echo "Compiling gaterecurrent2dnoind layer kernels by nvcc..."
nvcc -c -o gaterecurrent2dnoind_kernel.cu.o gaterecurrent2dnoind_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_61
cd ../../
python setup.py
