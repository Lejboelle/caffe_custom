#include <algorithm>
#include <vector>

#include "caffe/layers/triplet_loss_reg_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TripletLossRegLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  CHECK(bottom[0]->shape() == bottom[1]->shape())
      << "Inputs must have the same dimension.";
  CHECK(bottom[0]->shape() == bottom[2]->shape())
      << "Inputs must have the same dimension.";
  CHECK(bottom[0]->shape(1) == bottom[3]->shape(1))
      << "Inputs must have the same dimension.";
  CHECK(bottom[1]->shape(1) == bottom[4]->shape(1))
      << "Inputs must have the same dimension.";
  CHECK(bottom[2]->shape(1) == bottom[5]->shape(1))
      << "Inputs must have the same dimension.";
  int num_ = bottom[0]->num();
  int channels_ = bottom[0]->channels();
  int height_ = bottom[0]->height();
  int width_ = bottom[0]->width();
  diff_same_class_.Reshape(num_,channels_,height_,width_);
  diff_diff_class_.Reshape(num_,channels_,height_,width_);
  lab_anch_class_.Reshape(num_,channels_,height_,width_);
  lab_pos_class_.Reshape(num_,channels_,height_,width_);
  lab_neg_class_.Reshape(num_,channels_,height_,width_);
  
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
  batch_size_ = bottom[0]->shape(0);
  vec_dimension_ = bottom[0]->count() / batch_size_;
  vec_loss_.resize(batch_size_);
  vec_diff_.resize(batch_size_);
}

template <typename Dtype>
void TripletLossRegLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  TripletParameter param = this->layer_param_.triplet_param();
  
  alpha_ = param.alpha();
  gamma_ = param.gamma();
}

template <typename Dtype>
void TripletLossRegLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
											const vector<Blob<Dtype>*>& top) {
	
  int count = bottom[0]->count();
  caffe_sub(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(),
            diff_same_class_.mutable_cpu_data());
  caffe_sub(count, bottom[0]->cpu_data(), bottom[2]->cpu_data(),
            diff_diff_class_.mutable_cpu_data());
  caffe_sub(count, bottom[0]->cpu_data(), bottom[3]->cpu_data(),
			lab_anch_class_.mutable_cpu_data());
  caffe_sub(count, bottom[1]->cpu_data(), bottom[4]->cpu_data(),
			lab_pos_class_.mutable_cpu_data());
  caffe_sub(count, bottom[2]->cpu_data(), bottom[5]->cpu_data(),
			lab_neg_class_.mutable_cpu_data());

  Dtype loss = 0;
  for (int v = 0; v < batch_size_; ++v) {
	vec_diff_[v] = 
		caffe_cpu_dot(vec_dimension_,
                      lab_anch_class_.cpu_data() + v * vec_dimension_,
                      lab_anch_class_.cpu_data() + v * vec_dimension_) +
        caffe_cpu_dot(vec_dimension_,
                      lab_pos_class_.cpu_data() + v * vec_dimension_,
                      lab_pos_class_.cpu_data() + v * vec_dimension_) +
        caffe_cpu_dot(vec_dimension_,
                      lab_neg_class_.cpu_data() + v * vec_dimension_,
                      lab_neg_class_.cpu_data() + v * vec_dimension_);
	
    vec_loss_[v] =
        alpha_ +
        caffe_cpu_dot(vec_dimension_,
                      diff_same_class_.cpu_data() + v * vec_dimension_,
                      diff_same_class_.cpu_data() + v * vec_dimension_) -
        caffe_cpu_dot(vec_dimension_,
                      diff_diff_class_.cpu_data() + v * vec_dimension_,
                      diff_diff_class_.cpu_data() + v * vec_dimension_);
               
    vec_loss_[v] = std::max(Dtype(0), vec_loss_[v]) + gamma_*vec_diff_[v];
    loss += vec_loss_[v];
    
  }
  top[0]->mutable_cpu_data()[0] = loss / batch_size_;
}

template <typename Dtype>
void TripletLossRegLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down,
                                           const vector<Blob<Dtype>*>& bottom) {
  /*if (propagate_down[3]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  } */	
  const Dtype scale = Dtype(2) * top[0]->cpu_diff()[0] / bottom[0]->num();
  const int n = bottom[0]->count();

  caffe_sub(n, diff_same_class_.cpu_data(), diff_diff_class_.cpu_data(),
            bottom[0]->mutable_cpu_diff());
  caffe_cpu_axpby(n, -gamma_, lab_anch_class_.cpu_data(), Dtype(1),
				  bottom[0]->mutable_cpu_diff());
  caffe_scal(n, scale, bottom[0]->mutable_cpu_diff());
 
  caffe_copy(n, diff_same_class_.cpu_data(), bottom[1]->mutable_cpu_diff());
  caffe_axpy(n, -gamma_, lab_pos_class_.cpu_data(), bottom[1]->mutable_cpu_diff());
  caffe_scal(n, -scale, bottom[1]->mutable_cpu_diff());

  caffe_copy(n, diff_diff_class_.cpu_data(), bottom[2]->mutable_cpu_diff());
  caffe_axpy(n, -gamma_, lab_neg_class_.cpu_data(), bottom[2]->mutable_cpu_diff());
  caffe_scal(n, scale, bottom[2]->mutable_cpu_diff());
  
  ///// Old version ///////
  
  /*caffe_sub(n, diff_same_class_.cpu_data(), diff_diff_class_.cpu_data(),
            bottom[0]->mutable_cpu_diff());
  caffe_scal(n, scale, bottom[0]->mutable_cpu_diff());

  caffe_cpu_scale(n, -scale, diff_same_class_.cpu_data(),
                  bottom[1]->mutable_cpu_diff());

  caffe_cpu_scale(n, scale, diff_diff_class_.cpu_data(),
                  bottom[2]->mutable_cpu_diff()); */

  for (int v = 0; v < batch_size_; ++v) {
    if (vec_loss_[v] == 0) {
      caffe_set(vec_dimension_, Dtype(0),
                bottom[0]->mutable_cpu_diff() + v * vec_dimension_);
      caffe_set(vec_dimension_, Dtype(0),
                bottom[1]->mutable_cpu_diff() + v * vec_dimension_);
      caffe_set(vec_dimension_, Dtype(0),
                bottom[2]->mutable_cpu_diff() + v * vec_dimension_);
    }
  }
}

#ifdef CPU_ONLY
// STUB_GPU(TripletLossLayer);
#endif

INSTANTIATE_CLASS(TripletLossRegLayer);
REGISTER_LAYER_CLASS(TripletLossReg);

}  // namespace caffe
