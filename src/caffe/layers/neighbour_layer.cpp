#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/layers/neighbour_layer.hpp"

namespace caffe {
	

template <typename Dtype>
void NeighbourLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	
	CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
    << "corresponding to (num, channels, height, width)";
	CHECK_EQ(4, bottom[1]->num_axes()) << "Input must have 4 axes, "
	<< "corresponding to (num, channels, height, width)";
	CHECK_EQ(bottom[0]->num(),bottom[1]->num()) << "Number of images must"
	<< "be the same.";

	channels_i = bottom[0]->channels();
	channels_j = bottom[1]->channels();
	height_i = bottom[0]->height();
	height_j = bottom[1]->height();
	width_i = bottom[0]->width();
	width_j = bottom[1]->width();
	CHECK_EQ(height_i,height_j) << "Dimensions (height) must agree.";
	CHECK_EQ(width_i,width_j) << "Dimensions (width) must agree.";
	CHECK_EQ(channels_i,channels_j) << "Dimensions (channels) must agree.";
	top[0]->Reshape(bottom[0]->num(),channels_i,5*height_i,5*width_i); 
	top[1]->Reshape(bottom[0]->num(),channels_i,5*height_i,5*width_i);
}

template <typename Dtype>
void NeighbourLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	
	Dtype* top_data = top[0]->mutable_cpu_data();
	Dtype* top_data2 = top[1]->mutable_cpu_data();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	
	Dtype value_f = 0;
	Dtype value_f2 = 0;
	Dtype temp = 0;
	Dtype temp2 = 0;

	for (int n = 0; n < bottom[0]->num(); ++n){
		for (int c = 0; c < channels_i; ++c){
			for (int h = 0; h < height_i; ++h){
				const int index_h = 5*h;
				for (int w = 0; w < width_i; ++w){
					int count_y = 0;
					const int index_w = 5*w;
					value_f = bottom[0]->data_at(n,c,h,w);
					value_f2 = bottom[1]->data_at(n,c,h,w);

					for (int y = -2; y < 3; ++y){
						int count_x = 0;
						for (int x = -2; x < 3; ++x){
							int test1 = h+y;
							int test2 = w+x;
	
							//Check if at boundary
							if (test1 < 0 || test2 < 0 || test1 >= height_i || test2 >= width_i){
								temp = 0;
								temp2 = 0;
							}
							else{
								temp = bottom[1]->data_at(n,c,h+y,w+x);
								temp2 = bottom[0]->data_at(n,c,h+y,w+x);
							 }
							 top_data[top[0]->offset(n,c,index_h+count_y,index_w+count_x)] = value_f-temp;
							 top_data2[top[1]->offset(n,c,index_h+count_y,index_w+count_x)] = value_f2-temp2;
							count_x += 1;
						}	
						count_y += 1;
					}
				}
			}
		}
	} 
} 

template <typename Dtype>
void NeighbourLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


for (int i = 0; i < bottom.size(); ++i) {
  Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
  const int other = (i == 0) ? 1 : 0;
  
  for (int n = 0; n < top[i]->num(); ++n) {
	for (int c = 0; c < channels_i; ++c) {
		for (int h = 0; h < height_i; ++h) {
			const int index_h = 5*h;
			for (int w = 0; w < width_i; ++w) {
				int count = 0;
				const int index_w = 5*w;
				// Vers 1.0 - min error
				//Dtype min_val = top[0]->diff_at(n,c,5*h,5*w);
				//Dtype min_val2 = top[1]->diff_at(n,c,5*h,5*w);
				/*for (int y = 0; y < 5; ++y) {
					for (int x = 0; x < 5; ++x) {
						min_temp = top[i]->diff_at(n,c,(5*h)+y,(5*w)+x);
						min_temp2 = top[1]->diff_at(n,c,(5*h)+y,(5*w)+x);
						
						if (min_temp < min_val){
							min_val = min_temp;
						}
						if (min_temp2 < min_val2) {
							min_val2 = min_temp2;
						}
					}
				} 
				//bottom_diff[bottom[0]->offset(n,c,h,w)] = min_val;
				//bottom_diff2[bottom[1]->offset(n,c,h,w)] = min_val2; */
				
				// Vers. 2.0 - average error gradient w.r.t. bottom	
				for (int y = 0; y < 5; ++y){
					for (int x = 0; x < 5; ++x){
						count += 1;
						bottom_diff[bottom[i]->offset(n,c,h,w)] += top[i]->diff_at(n,c,index_h+y,index_w+x);
						if ((x-2)+w >= width_i || (y-2)+h >= height_i || (x-2)+w < 0 || (y-2)+h < 0){
							continue;
						}
						else{
							int off_y = 2-(y-2);
							int off_x = 2-(x-2);
							int H = 5*((y-2)+h)+off_y;
							int W = 5*((x-2)+w)+off_x;
							bottom_diff[bottom[i]->offset(n,c,h,w)] += (-1*top[other]->diff_at(n,c,H,W));
							count += 1;	
						}
					}
				} 
				
				bottom_diff[bottom[i]->offset(n,c,h,w)] /= count;
			}
		}
	}
  }
 }	
}

#ifdef CPU_ONLY
STUB_GPU(NeighbourLayer);
#endif

INSTANTIATE_CLASS(NeighbourLayer);
REGISTER_LAYER_CLASS(Neighbour);

} // namespace caffe
