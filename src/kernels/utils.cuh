// #ifndef UTILS_CUH__
// #define UTILS_CUH__
#include <cuda_runtime.h>


class CUDA_Property_List{
    //c++ wrapper for list of CUDA properties
    private:
        int numDevices_;
        cudaDeviceProp* propertyList_;

    public:
        CUDA_Property_List();
        ~CUDA_Property_List();

        void print_properties();

        inline cudaDeviceProp get_properties(int i){
            return propertyList_[i];
        }
        inline void set_properties(cudaDeviceProp prop, int i){
            propertyList_[i] = prop;
        }
        inline int get_num_devices(){
            return numDevices_;
        }

};

void get_device_properties();

// #endif

