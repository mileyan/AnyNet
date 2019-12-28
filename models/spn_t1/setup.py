from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

setup(
    name='gaterecurrent2dnoind_cuda',
    ext_modules=[
        CUDAExtension('gaterecurrent2dnoind_cuda', [
            'src/gaterecurrent2dnoind_cuda.cpp',
            'src/gaterecurrent2dnoind_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })



# this_file = os.path.dirname(__file__)
#
# sources = []
# headers = []
# defines = []
# with_cuda = False
#
# if torch.cuda.is_available():
#     print('Including CUDA code.')
#     sources += ['src/gaterecurrent2dnoind_cuda.c']
#     headers += ['src/gaterecurrent2dnoind_cuda.h']
#     defines += [('WITH_CUDA', None)]
#     with_cuda = True
#
# this_file = os.path.dirname(os.path.realpath(__file__))
# extra_objects = ['src/cuda/gaterecurrent2dnoind_kernel.cu.o']
# extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]
#
# ffi = create_extension(
#     '_ext.gaterecurrent2dnoind',
#     headers=headers,
#     sources=sources,
#     define_macros=defines,
#     relative_to=__file__,
#     with_cuda=with_cuda,
#     extra_objects=extra_objects
# )
#
# if __name__ == '__main__':
#     ffi.build()
