#include <vector>
#include <cmath>
#include "caffe/layers/normalization_layer.hpp"
#include "caffe/util/math_functions.hpp"
 
namespace caffe {
 
template <typename Dtype> 
void NormalizationLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    this->Forward_cpu(bottom, top);
}   
 
template <typename Dtype>
void NormalizationLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, 
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    this->Backward_cpu(top, propagate_down, bottom);
}
INSTANTIATE_LAYER_GPU_FUNCS(NormalizationLayer);
} // namespace caffe