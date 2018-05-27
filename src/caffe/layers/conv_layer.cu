#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

vector <float *> pruning1; //used for entry matrix calculation
vector <float *> pruning2;
int epoch=0; //counts no of epochs
int images=0; //counts no of images fed to neural net
int Z=1; //skipping interval.
bool norr=true;
int conv_batches=0; //counter to track no of mini-batches that are fed to conv layer
int N1=5; //size of mega-batch or no of mini-batches in a mega-batch

template <typename Dtype>
__global__ void Threshold_pruning(const int n,
    Dtype* in, float* out) {
  CUDA_KERNEL_LOOP(index, n) {
	  if (in[index]>10) //threshold=10
		  out[index] =1234; //some magic number for me to identify if a neuron is marked for skipping computation or not based on threshold value.
	  else
		  out[index] = in[index];
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int p_t =0;  //pre-stan interval
  const Dtype* weight = this->blobs_[0]->gpu_data();
  if(this->phase_==TRAIN && this->top_dim_==11520){
       ++conv_batches;
  }
  int indexx=(conv_batches%N1)-1;
  if (indexx<0)
         indexx=N1-1;
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    const int count = this->top_dim_;
    //this->num_ is no of images in mini-batch
    if (epoch==0 && images< N1*this->num_ && this->phase_ == TRAIN){ //repeat this until we allocate memory for entry matrix for all neurons for all images in megabatch.
	    float* d_x = NULL;
	    cudaMalloc(&d_x, sizeof(float)*count*this->num_);
	    cudaMemset(d_x,0,count*this->num_*sizeof(float));
	    if (count == 11520) //output dimension of LeNet are 11520 for 1st layer and 3200 for 2nd layer.
	         pruning1.push_back(d_x);
	    else
        	 pruning2.push_back(d_x);
    }
    if (conv_batches%(N1*(Z+1)) <= N1 && conv_batches%(N1*(Z+1))>0) {
              norr=true;
    } else {
              norr=false;
    }
    for (int n = 0; n < this->num_; ++n) {
      if (this->phase_ == TEST || ((this->phase_ == TRAIN) && (epoch<p_t))){
          this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
              top_data + n * this->top_dim_, NULL,true);
      } else {
	  if (norr){
		this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
	              top_data + n * this->top_dim_,NULL,true);
		if (count == 11520)
			 Threshold_pruning<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, top_data + n * this->top_dim_, pruning1[indexx]+n * this->top_dim_);
		else 
			 Threshold_pruning<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, top_data + n * this->top_dim_, pruning2[indexx]+n * this->top_dim_);
		CUDA_POST_KERNEL_CHECK;
	  } else {
		if (count == 11520)
			this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight, top_data + n * this->top_dim_, pruning1[indexx]+n * this->top_dim_, false);
		else 
			this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight, top_data + n * this->top_dim_, pruning2[indexx]+n * this->top_dim_, false); 
	  }
      }
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
    if (this->phase_ == TRAIN){
	if(this->top_dim_ == 3200) //dimension of last layer
		images+=this->num_;
	if (images >=60000){ //MNIST has 60k images in total
		epoch++;
		images=0;
	}
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
