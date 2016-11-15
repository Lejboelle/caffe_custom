#include <algorithm>
#include <cfloat>
#include <vector>
#include <iostream>
#include <stdio.h>

#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/neighbour_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void NeighbourLayerForward(const int n_threads, const int count,
	const int channels, const int height, const int width,
    const Dtype* in1, const Dtype* in2, Dtype* out1, Dtype* out2, Dtype temp1, Dtype temp2) {
  CUDA_KERNEL_LOOP(index, count) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;

	const Dtype* bot_slice1 = in1 + (n * channels + c) * height * width;
	const Dtype* bot_slice2 = in2 + (n * channels + c) * height * width;
	Dtype value_f1 = bot_slice1[h*width+w];
	Dtype value_f2 = bot_slice2[h*width+w];	

    for (int y = -2; y < 3; ++y){
		for (int x = -2; x < 3; ++x){
			int test1 = h+y;
			int test2 = w+x;

			if (test1 < 0 || test2 < 0 || test1 >= height || test2 >= width){
				temp1 = 0;
				temp2 = 0;
			}
			else{
				temp1 = bot_slice1[(h+y)*width+w+x];
				temp2 = bot_slice2[(h+y)*width+w+x];				
				
			}
			int index_out = ((n * channels + c) * (5*height) + (5*h) + y + 2) * (5*width) + (5*w) + x + 2;
			out1[index_out] = (value_f1-temp2);
			out2[index_out] = (value_f2-temp1);
		}
	}
  }
}

template <typename Dtype>
void NeighbourLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_data2 = bottom[1]->gpu_data();

  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* top_data2 = top[1]->mutable_gpu_data();
  int count = top[0]->count();
  int count_bottom = bottom[0]->count();
  Dtype temp1 = 0;
  Dtype temp2 = 0;
  // NOLINT_NEXT_LINE(whitespace/operators)
  NeighbourLayerForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, count_bottom, channels_i,
      height_i, width_i, bottom_data, bottom_data2, top_data, top_data2, temp1, temp2);
  CUDA_POST_KERNEL_CHECK;
}	
	
template <typename Dtype>
__global__ void NeighbourLayerBackward(const int n_threads, const int count,
	const int channels, const int height, const int width,
    const Dtype* const in1, const Dtype* const in2, Dtype* const out1, Dtype temp1, Dtype temp2, int other) {
  CUDA_KERNEL_LOOP(index, count) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
	
    //Dtype min_val1 = in1[((n * channels + c) * (5*height) + (5*h)) * (5*width) + (5*w)];
    //Dtype min_val2 = in2[((n * channels + c) * (5*height) + (5*h)) * (5*width) + (5*w)];	

    // Vers. 1.0
   /* for (int y = 0; y < 5; ++y){
		for (int x = 0; x < 5; ++x){
			temp1 = in1[((n * channels + c) * (5*height) + (5*h+y)) * (5*width) + (5*w+x)];
			temp2 = in2[((n * channels + c) * (5*height) + (5*h+y)) * (5*width) + (5*w+x)];
			
			if (temp1 < min_val1){
				min_val1 = temp1;
			}
			if (temp2 < min_val2){
				min_val2 = temp2;
			}
		}
	} */ 
	
	int count = 0;
	int index_o = ((n * channels + c) * height + h) * width + w;
	
	// Vers 2.0
		for (int y = 0; y < 5; ++y){
			for (int x = 0; x < 5; ++x){
				count += 1;
				out1[index_o] += in1[((n * channels + c) * (5*height) + (5*h) + y) * 5*width + (5*w) + x];
				if ((x-2)+w >= width || (y-2)+h >= height || (x-2)+w < 0 || (y-2)+h < 0){
					continue;
				}
				else{
					int off_y = 2-(y-2);
					int off_x = 2-(x-2);
					out1[index_o] += -1*in2[((n * channels + c) * (5*height) + 5*((y-2)+h)+off_y) * (5*width) + 5*((x-2)+w) + off_x];
					count += 1;
				}
			}
		} 
		
	out1[index_o] /= count;
	/*if (other == 1){
	out1[index_o] = min_val1;
	}
	else{
	out1[index_o] = min_val2;
	}*/
  }
}

template <typename Dtype>
void NeighbourLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  for (int i = 0; i < bottom.size(); ++i){
	  const int other = (i == 0) ? 1 : 0;
	  
	  const Dtype* top_diff = top[i]->gpu_diff();
	  const Dtype* top_diff2 = top[other]->gpu_diff();
	  Dtype* bottom_data = bottom[i]->mutable_gpu_diff();
	  
	  const int count = top[i]->count();
	  const int bottom_count = bottom[i]->count();
	  Dtype temp1 = 0;
	  Dtype temp2 = 0;

	  // NOLINT_NEXT_LINE(whitespace/operators)
	  NeighbourLayerBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
		  count, bottom_count, channels_i,
		  height_i, width_i, top_diff, top_diff2, bottom_data, temp1, temp2, other);
	  CUDA_POST_KERNEL_CHECK;
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(NeighbourLayer);

} // namespace caffe

