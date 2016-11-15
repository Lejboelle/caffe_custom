#ifndef CAFFE_TRIPLET_LOSS_LAYER_HPP_
#define CAFFE_TRIPLET_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class TripletLossRegLayer : public LossLayer<Dtype> {
 public:
  explicit TripletLossRegLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}

  void Reshape(const vector<Blob<Dtype>*>& bottom,
               const vector<Blob<Dtype>*>& top);

  inline const char* type() const { return "TripletLossReg"; }
  inline int ExactNumBottomBlobs() const { return 6; }
  
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  

 protected:
  /// @copydoc TripletLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  
  Blob<Dtype> diff_same_class_;
  Blob<Dtype> diff_diff_class_;
  Blob<Dtype> lab_anch_class_;
  Blob<Dtype> lab_pos_class_;
  Blob<Dtype> lab_neg_class_;
  Dtype alpha_;
  Dtype gamma_;
  vector<Dtype> vec_loss_;
  vector<Dtype> vec_diff_;
  int batch_size_;
  int vec_dimension_;
};

}  // namespace caffe

#endif  // CAFFE_TRIPLET_LOSS_LAYER_HPP_
