#include "utils.cuh"
#include <cuda_runtime.h>
#include <iostream>

CUDA_Property_List::CUDA_Property_List() {
    cudaGetDeviceCount( &numDevices_);
    propertyList_ = new cudaDeviceProp[numDevices_];
    for (int i = 0; i < numDevices_; i++) {
        cudaGetDeviceProperties( &propertyList_[i], i);
    }
}

CUDA_Property_List::~CUDA_Property_List(){
    delete[] propertyList_;
}


void CUDA_Property_List::print_properties(){
    std::cout<<"number of Devices: "<<numDevices_<<'\n';
    for (int i = 0; i < numDevices_; i++) {
        std::cout<<"\tDevice: "<<i<<'\n';
        std::cout<<"\tmaxThreadsPerBlock: "<<propertyList_[i].maxThreadsPerBlock<<'\n';
        std::cout<<"\tsharedMemoryPerBlock: "<<propertyList_[i].sharedMemPerBlock<<" Bytes\n";

    }
}



void get_device_properties(){
    CUDA_Property_List myList = CUDA_Property_List();
    myList.print_properties();        
    return;
}