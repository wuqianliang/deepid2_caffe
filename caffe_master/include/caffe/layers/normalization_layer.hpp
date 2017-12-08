// 主要实现了feature的归一化
#ifndef CAFFE_NORMALIZATION_LAYER_HPP_
#define CAFFE_NORMALIZATION_LAYER_HPP_
 
#include <vector>
 
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
 
#include "caffe/layers/neuron_layer.hpp"
 
namespace caffe {
 
template <typename Dtype>
class NormalizationLayer : public NeuronLayer<Dtype> {
 public:
  explicit NormalizationLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "Normalization"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  Blob<Dtype> norm_val_; // 记录每个feature的模
};
 
}  // namespace caffe
 
#endif  // CAFFE_NORMALIZATION_LAYER_HPP_