# README #

## When compiling outside of CLion, copy the following code into the CMakeCXXCompiler.cmake file:

    set(CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES 
    "/calculate/gabriel/software/dealii-9.2.0/opt/spack/linux-ubuntu18.04-x86_64/gcc-9.3.0/openmpi-3.1.6-5ajbpkwxzfdutovyb4g74biinrezkagl/include;
    /calculate/gabriel/software/dealii-9.2.0/opt/spack/linux-ubuntu18.04-x86_64/gcc-7.4.0/gcc-9.3.0-kg3o7qkgb7zyj4rd3n3rvnvt53fzsjpb/include/c++/9.3.0;
    /calculate/gabriel/software/dealii-9.2.0/opt/spack/linux-ubuntu18.04-x86_64/gcc-7.4.0/gcc-9.3.0-kg3o7qkgb7zyj4rd3n3rvnvt53fzsjpb/include/c++/9.3.0/x86_64-pc-linux-gnu;
    /calculate/gabriel/software/dealii-9.2.0/opt/spack/linux-ubuntu18.04-x86_64/gcc-7.4.0/gcc-9.3.0-kg3o7qkgb7zyj4rd3n3rvnvt53fzsjpb/include/c++/9.3.0/backward;
    /calculate/gabriel/software/dealii-9.2.0/opt/spack/linux-ubuntu18.04-x86_64/gcc-7.4.0/gcc-9.3.0-kg3o7qkgb7zyj4rd3n3rvnvt53fzsjpb/lib/gcc/x86_64-pc-linux-gnu/9.3.0/include;
    /usr/local/include;
    /calculate/gabriel/software/dealii-9.2.0/opt/spack/linux-ubuntu18.04-x86_64/gcc-7.4.0/gcc-9.3.0-kg3o7qkgb7zyj4rd3n3rvnvt53fzsjpb/include;
    /calculate/gabriel/software/dealii-9.2.0/opt/spack/linux-ubuntu18.04-x86_64/gcc-7.4.0/gcc-9.3.0-kg3o7qkgb7zyj4rd3n3rvnvt53fzsjpb/lib/gcc/x86_64-pc-linux-gnu/9.3.0/include-fixed;
    /usr/include/x86_64-linux-gnu;
    /usr/include")

    set(CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES 
    "/calculate/gabriel/software/dealii-9.2.0/opt/spack/linux-ubuntu18.04-x86_64/gcc-9.3.0/hwloc-1.11.11-e37dfy4wajxpzafufrdze4kmbdbdiggs/lib;
    /calculate/gabriel/software/dealii-9.2.0/opt/spack/linux-ubuntu18.04-x86_64/gcc-9.3.0/zlib-1.2.11-cqck4xltqqtchm2vkfz3eatofh7qhbeq/lib;
    /calculate/gabriel/software/dealii-9.2.0/opt/spack/linux-ubuntu18.04-x86_64/gcc-9.3.0/openmpi-3.1.6-5ajbpkwxzfdutovyb4g74biinrezkagl/lib;
    /calculate/gabriel/software/dealii-9.2.0/opt/spack/linux-ubuntu18.04-x86_64/gcc-7.4.0/gcc-9.3.0-kg3o7qkgb7zyj4rd3n3rvnvt53fzsjpb/lib/gcc/x86_64-pc-linux-gnu/9.3.0;
    /calculate/gabriel/software/dealii-9.2.0/opt/spack/linux-ubuntu18.04-x86_64/gcc-7.4.0/gcc-9.3.0-kg3o7qkgb7zyj4rd3n3rvnvt53fzsjpb/lib64;
    /lib/x86_64-linux-gnu;
    /lib64;
    /usr/lib/x86_64-linux-gnu;
    /calculate/gabriel/software/dealii-9.2.0/opt/spack/linux-ubuntu18.04-x86_64/gcc-7.4.0/gcc-9.3.0-kg3o7qkgb7zyj4rd3n3rvnvt53fzsjpb/lib")