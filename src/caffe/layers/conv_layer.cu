#include <vector>
#include <cuda_fp16.h>
#include <math.h>
#include "/usr/local/cuda/include/math_constants.h"
#include "caffe/layers/conv_layer.hpp"
#include <cuda_profiler_api.h>

namespace caffe {

float* pruning1; 
float* pruning2;
float* pruning3;
float* op1;
float* op2;
float* op3;
float* first1;
float* first2;
float* first3;
float * ruing1;
float * ruing2;
float * ruing3;
float* mu1;
float* mu2;
float* mu3;
float* sig1;
float* sig2;
float* sig3;
int epoch=0; //counts no of epochs
int images=0; //counts no of images fed to neural net
int Z=2; //skipping interval.
bool norr=true;
int conv_batches=0; //counter to track no of mini-batches that are fed to conv layer
int N1=2; //size of mega-batch or no of mini-batches in a mega-batch
bool init=true;
int epoch_int=0;

template <typename Dtype>
__global__ void update1(const int n, float* sig, float* op, float* mu, const int epoch, Dtype* curr_op, float* pruning, const int idx, float* first) {
	CUDA_KERNEL_LOOP(index, n) {
		index+=idx;
		float c_op=curr_op[index-idx];
		float c_sig=sig[index];
		float c_pun=pruning[index];
		float old_mu=mu[index];
		float c_fst=first[index];
		float opp=op[index];
        	c_sig+=(epoch/(epoch+1.0))*(c_op-old_mu)*(c_op-old_mu);
                float muu=(epoch/(epoch+1.0))*old_mu+(c_op/(epoch+1));
		if (!epoch)
			op[index]=0;
		else
	                opp+=((epoch-1)/((epoch+1)*(epoch+1)))*(c_op-old_mu)*(c_op-old_mu)+((c_pun-muu)*(c_op-muu))+((c_op-old_mu)/(epoch+1))*(c_pun+c_fst-(2*old_mu));
		pruning[index]=c_op;
		sig[index]=c_sig;
		op[index]=opp;
		mu[index]=muu;
		index-=idx;
	}
}
template <typename Dtype>
__global__ void update(const int n, float* sig, float* op, float* mu, const int epoch, Dtype* curr_op, float* pruning, const int idx, float* ruing, float* first, bool init) {
	CUDA_KERNEL_LOOP(index, n) {
		index+=idx;
		float c_op=curr_op[index-idx];
		float old_mu=init?0:mu[index];
		float c_sig=init?0:sig[index];
		float opp=init?0:op[index];
		float c_prun=init?0:pruning[index];
		if (init)
			first[index]=c_op;
		float muu=(epoch/(epoch+1.0))*old_mu+(c_op/(epoch+1));
        	c_sig+=(epoch/(epoch+1.0))*(c_op-old_mu)*(c_op-old_mu);
		if (epoch)
	        	opp+=((epoch-1)/((epoch+1)*(epoch+1)))*(c_op-old_mu)*(c_op-old_mu)+((c_prun-muu)*(c_op-muu))+((c_op-old_mu)/(epoch+1))*(c_prun+first[index]-(2*old_mu));
		else
			opp=0.0;
		if (c_sig==0)
                	ruing[index]=c_op;
                else
                        ruing[index]=abs(opp/c_sig)<0.1?1234:c_op; 
		pruning[index]=c_op;
		sig[index]=c_sig;
		op[index]=opp;
		mu[index]=muu;
		index-=idx;
	}
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int p_t =2;  //pre-stan interval
  const Dtype* weight = this->blobs_[0]->gpu_data();
  int indexx,temp_var;
  if(this->phase_==TRAIN && this->top_dim_==32768){
       ++conv_batches;
  }
  indexx=(conv_batches-1)%N1;
  if(this->phase_==TRAIN && this->top_dim_==32768){
	  if (!indexx)
		epoch_int++;
  }
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    const int count = this->top_dim_;
    if (epoch==0 && images==0 && this->phase_ == TRAIN){
            if (count == 32768){//11520) {
                cudaMalloc((void **)&pruning1,count*N1*this->num_*sizeof(float)); //saves prev epoch output
                cudaMemset(pruning1,0,count*N1*this->num_*sizeof(float));
                cudaMalloc((void **)&mu1,count*N1*this->num_*sizeof(float)); //running avg
                cudaMemset(mu1,0,count*N1*this->num_*sizeof(float));
                cudaMalloc((void **)&sig1,count*N1*this->num_*sizeof(float)); //running std dev
                cudaMemset(sig1,0,count*N1*this->num_*sizeof(float));
                cudaMalloc((void **)&op1,count*N1*this->num_*sizeof(float)); //num of autocorrelation
                cudaMemset(op1,0,count*N1*this->num_*sizeof(float));
                cudaMalloc((void **)&ruing1,count*N1*this->num_*sizeof(float)); //entry matrix
                cudaMemset(ruing1,0,count*N1*this->num_*sizeof(float));
                cudaMalloc((void **)&first1,count*N1*this->num_*sizeof(float)); //first value of the series
                cudaMemset(first1,0,count*N1*this->num_*sizeof(float));
            } else if  (count==8192){
                cudaMalloc((void **)&pruning2,count*N1*this->num_*sizeof(float)); 
                cudaMemset(pruning2,0,count*N1*this->num_*sizeof(float));
                cudaMalloc((void **)&mu2,count*N1*this->num_*sizeof(float)); 
                cudaMemset(mu2,0,count*N1*this->num_*sizeof(float));
                cudaMalloc((void **)&sig2,count*N1*this->num_*sizeof(float));
                cudaMemset(sig2,0,count*N1*this->num_*sizeof(float));
                cudaMalloc((void **)&op2,count*N1*this->num_*sizeof(float)); 
                cudaMemset(op2,0,count*N1*this->num_*sizeof(float));
                cudaMalloc((void **)&ruing2,count*N1*this->num_*sizeof(float));
                cudaMemset(ruing2,0,count*N1*this->num_*sizeof(float));
                cudaMalloc((void **)&first2,count*N1*this->num_*sizeof(float)); 
                cudaMemset(first2,0,count*N1*this->num_*sizeof(float));
            }else{
                cudaMalloc((void **)&pruning3,count*N1*this->num_*sizeof(float));
                cudaMemset(pruning3,0,count*N1*this->num_*sizeof(float));
                cudaMalloc((void **)&mu3,count*N1*this->num_*sizeof(float));
                cudaMemset(mu3,0,count*N1*this->num_*sizeof(float));
                cudaMalloc((void **)&sig3,count*N1*this->num_*sizeof(float));
                cudaMemset(sig3,0,count*N1*this->num_*sizeof(float));
                cudaMalloc((void **)&op3,count*N1*this->num_*sizeof(float));
                cudaMemset(op3,0,count*N1*this->num_*sizeof(float));
                cudaMalloc((void **)&ruing3,count*N1*this->num_*sizeof(float));
                cudaMemset(ruing3,0,count*N1*this->num_*sizeof(float));
                cudaMalloc((void **)&first3,count*N1*this->num_*sizeof(float));
                cudaMemset(first3,0,count*N1*this->num_*sizeof(float));
            }
    }
    if(this->phase_ == TRAIN ){
	    temp_var=conv_batches%(N1*(Z+1));
	    if (temp_var<=2*N1 && temp_var>0) {
	              norr=true;
		      init=(temp_var<=N1)?true:false;
	    } else {
              	      norr=false;
        	      init=false;
    	    }
    }
    for (int n = 0; n < this->num_; ++n) {
      if (this->phase_ == TEST || ((this->phase_ == TRAIN) && (epoch<p_t))){ 
          this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
              top_data + n * this->top_dim_, NULL,true);
      } else {
	  if (norr){
		this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
	              top_data + n * this->top_dim_,NULL,true);
                        if (count == 32768){
				update<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, sig1, op1, mu1, epoch_int-1, 
					top_data + n * this->top_dim_, pruning1, indexx*this->num_*this->top_dim_+(n*this->top_dim_), ruing1, first1, init);
                        } else if (count==8192) {
				update<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, sig2, op2, mu2, epoch_int-1, 
					top_data + n * this->top_dim_, pruning2, indexx*this->num_*this->top_dim_+(n*this->top_dim_), ruing2, first2, init);
                        } else {
				update<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, sig3, op3, mu3, epoch_int-1, 
					top_data + n * this->top_dim_, pruning3, indexx*this->num_*this->top_dim_+(n*this->top_dim_), ruing3, first3, init);
                        }
		CUDA_POST_KERNEL_CHECK;
	  } else {
                        if (count == 32768){
				this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight, 
					top_data + n * this->top_dim_, ruing1+(indexx*this->num_*this->top_dim_)+(n*this->top_dim_), false);
				update1<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, sig1, op1, mu1, epoch_int-1, 
					top_data + n * this->top_dim_, pruning1, indexx*this->num_*this->top_dim_+(n*this->top_dim_), first1);
                        } else if (count==8192) {
				this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight, 
					top_data + n * this->top_dim_, ruing2+(indexx*this->num_*this->top_dim_)+(n*this->top_dim_), false);
				update1<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, sig2, op2, mu2, epoch_int-1, 
					top_data + n * this->top_dim_, pruning2, indexx*this->num_*this->top_dim_+(n*this->top_dim_), first2);
                        } else {
				this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight, 
					top_data + n * this->top_dim_, ruing3+(indexx*this->num_*this->top_dim_)+(n*this->top_dim_), false);
				update1<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, sig3, op3, mu3, epoch_int-1, 
					top_data + n * this->top_dim_, pruning3, indexx*this->num_*this->top_dim_+(n*this->top_dim_), first3);
                        }
                CUDA_POST_KERNEL_CHECK;
	  }
      }
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
    if (this->phase_ == TRAIN){
    	if (!temp_var && this->top_dim_==4096){
		epoch_int=0;
    	}
	if(this->top_dim_ == 4096 & norr) //dimension of last layer
		images+=this->num_;
	if (images >=50000 && conv_batches%(N1*(Z+1))==0){ //MNIST has 60k images in total
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
