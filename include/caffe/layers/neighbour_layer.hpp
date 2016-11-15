#ifndef CAFFE_NEIGHBOUR_LAYER_HPP_
#define CAFFE_NEIGHBOUR_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


template <typename Dtype>
class NeighbourLayer : public Layer<Dtype> {
 public:
 explicit NeighbourLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
     
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
      
  virtual inline const char* type() const { return "Neighbour"; }
 
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 2; }
  
  protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int channels_i, channels_j;
  int height_i, width_i;
  int height_j, width_j;
};

}  // namespace caffe

#endif  // CAFFE_NEIGHBOUR_LAYER_HPP_
