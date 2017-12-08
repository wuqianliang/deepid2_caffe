#include <vector>
#include <cmath>
#include "caffe/layers/normalization_layer.hpp"
#include "caffe/util/math_functions.hpp"
 
namespace caffe {
 
template <typename Dtype>
void NormalizationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    NeuronLayer<Dtype>::LayerSetUp(bottom, top);
    CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
        "allow in-place computation.";
    norm_val_.Reshape(bottom[0]->shape(0), 1, 1, 1); // 申请norm的内存
}
 
 
template <typename Dtype> 
void NormalizationLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
 
    Dtype *norm_val_cpu_data = norm_val_.mutable_cpu_data();
    for (int n = 0; n < bottom[0]->shape(0); ++ n) {
        // 计算每个c * h * w的区域的模
        norm_val_cpu_data[n] = std::sqrt(static_cast<float>(
                    caffe_cpu_dot<Dtype>(
                        bottom[0]->count(1), 
                        bottom[0]->cpu_data() + bottom[0]->offset(n), 
                        bottom[0]->cpu_data() + bottom[0]->offset(n)
                        )
                    ));
        // 将每个bottom归一化，输出到top
        caffe_cpu_scale<Dtype>(
                top[0]->count(1), 
                1. / norm_val_cpu_data[n], 
                bottom[0]->cpu_data() + bottom[0]->offset(n), 
                top[0]->mutable_cpu_data() + top[0]->offset(n)
                );
    }
}
 
template <typename Dtype>
void NormalizationLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, 
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    
    const Dtype *norm_val_cpu_data = norm_val_.cpu_data();
    const Dtype *top_diff = top[0]->cpu_diff();
    Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype *bottom_data = bottom[0]->cpu_data();
 
    caffe_copy(top[0]->count(), top_diff, bottom_diff);
    
    for (int n = 0; n < top[0]->shape(0); ++ n) {
        Dtype a = - 1./(norm_val_cpu_data[n] * norm_val_cpu_data[n] * norm_val_cpu_data[n]) * caffe_cpu_dot<Dtype>(
                top[0]->count(1),
                top_diff + top[0]->offset(n),
                bottom_data + bottom[0]->offset(n)
                );
        Dtype b = 1. / norm_val_cpu_data[n];
        caffe_cpu_axpby<Dtype>(
                top[0]->count(1),
                a,
                bottom_data + bottom[0]->offset(n),
                b,
                bottom_diff + top[0]->offset(n)
                );
    }
}
#ifdef CPU_ONLY
STUB_GPU(NormalizationLayer);
#endif
 
INSTANTIATE_CLASS(NormalizationLayer);
REGISTER_LAYER_CLASS(Normalization);
 
} // namespace caffe