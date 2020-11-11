# Delay and sum GPU implementation with Matlab, CUDA and C++ MEX API

## Description
This example includes a CUDA implementation of a conventiona delay-and-sum
algorithm for plane-wave US acquisitions with various different transmit steering angles. 
The delay-and-sum written in c++ using CUDA for NVIDIA GPUs and is accessed from MATLAB via the [C++ MEX API](https://ch.mathworks.com/help/matlab/cpp-mex-file-applications.html). 

## Requrements

### Window 10
This implementation was tested with Matlab R2019a under Windows 10. To successfully compile the c++ source file, Visual Studio 2017 and CUDA V10.0 need to be installed. 
The path of MW_NVCC_PATH can be set by calling setenv('MW_NVCC_PATH','PATH_TO_CUDA\v10.0\bin') in Matlab.

