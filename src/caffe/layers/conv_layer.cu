#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

float * pruning1; //used for entry matrix calculation
float * pruning2;
float * pruning3;
float * op1;
float * op2;
float * op3;
float * ruing1;
float * ruing2;
float * ruing3;
float * mu1;
float * mu2;
float * mu3;
float * sig1;
float * sig2;
float * sig3;
int epoch=0; //counts no of epochs
int images=0; //counts no of images fed to neural net
int Z=1; //skipping interval.
bool norr=true;
int conv_batches=0; //counter to track no of mini-batches that are fed to conv layer
int N1=2; //size of mega-batch or no of mini-batches in a mega-batch

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
__global__ void Threshold_pruning(const int n,
    float* in, float* out, Dtype* out2) {
  CUDA_KERNEL_LOOP(index, n) {
          if (in[index]<0.5){
                  out[index] =1234;
          } else {
                  out[index] = out2[index];
          }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int p_t =2;  //pre-stan interval
  const Dtype* weight = this->blobs_[0]->gpu_data();
  if(this->phase_==TRAIN && this->top_dim_==32768){
       ++conv_batches;
  }
  int indexx=(conv_batches%N1)-1;
  if (indexx<0)
         indexx=N1-1;
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    const int count = this->top_dim_;
#if 0
    if (epoch==0 && images< N1*this->num_ && this->phase_ == TRAIN){ //repeat this until we allocate memory for entry matrix for all neurons for all images in megabatch.
	    float* d_x = NULL;
	    cudaMalloc(&d_x, sizeof(float)*count*this->num_);
	    cudaMemset(d_x,0,count*this->num_*sizeof(float));
	    if (count == 32768) //output dimension of LeNet are 11520 for 1st layer and 3200 for 2nd layer.
	         pruning1.push_back(d_x);
	    else if (count == 8192)
        	 pruning2.push_back(d_x);
	    else
		 pruning3.push_back(d_x);
    }
#endif
    if (epoch==0 && images==0 && this->phase_ == TRAIN){
            float* d_x = (float*)calloc(count*N1*this->num_, sizeof(float));
            float* mu_x = (float*)calloc(count*N1*this->num_, sizeof(float));
            float* sig_x = (float*)calloc(count*N1*this->num_, sizeof(float));
            float* num = (float*)calloc(count*N1*this->num_, sizeof(float));
            if (count == 32768){//11520) {
                pruning1=d_x; //saves prev epoch output
                mu1=mu_x; //running avg
                sig1=sig_x; //running std dev
                op1=num; //num of autocorrelation
                cudaMalloc((void **)&ruing1,count*N1*this->num_*sizeof(float)); //entry matrix
                cudaMemset(ruing1,0,count*N1*this->num_*sizeof(float));
            } else if  (count==8192){
                pruning2=d_x;
                mu2=mu_x;
                op2=num;
                sig2=sig_x;
                cudaMalloc((void **)&ruing2,count*N1*this->num_*sizeof(float));
                cudaMemset(ruing2,0,count*N1*this->num_*sizeof(float));
            }else{
                pruning3=d_x;
                mu3=mu_x;
                op3=num;
                sig3=sig_x;
                cudaMalloc((void **)&ruing3,count*N1*this->num_*sizeof(float));
                cudaMemset(ruing3,0,count*N1*this->num_*sizeof(float));
            }
    }

    if (conv_batches%(N1*(Z+1)) <= N1 && conv_batches%(N1*(Z+1))>0) {
              norr=true;
    } else {
              norr=false;
    }
    float new_val[count] = {0};
    for (int n = 0; n < this->num_; ++n) {
      float* curr_op=(float *)malloc(count*sizeof(float));
      float* gaf=NULL;
      if (this->phase_ == TEST || ((this->phase_ == TRAIN) && (epoch<p_t))){
          this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
              top_data + n * this->top_dim_, NULL,true);
	  if (this->phase_ == TRAIN){
                        cudaDeviceSynchronize();
                        cudaMemcpy(curr_op,top_data + n * this->top_dim_,this->top_dim_,cudaMemcpyDeviceToHost);
                        //update mean and var now.
			int index=indexx*this->num_*this->top_dim_+(n*this->top_dim_);
                        for (int neuron=0; neuron<count; neuron++){
                                index+=neuron;
                                if (count==32768){
                                        sig1[index]+=(epoch/(epoch+1.0))*(curr_op[neuron]-mu1[index])*(curr_op[neuron]-mu1[index]);
                                        op1[index]+=(epoch/(epoch+1.0))*(curr_op[neuron]-mu1[index])*(pruning1[index]-mu1[index]);
                                        mu1[index]= (epoch/(epoch+1.0))*mu1[index]+(curr_op[neuron]/(epoch+1));
                                } else if (count==8192){
                                        op2[index]+=(epoch/(epoch+1.0))*(curr_op[neuron]-mu2[index])*(pruning2[index]-mu2[index]);
                                        sig2[index]+=(epoch/(epoch+1.0))*(curr_op[neuron]-mu2[index])*(curr_op[neuron]-mu2[index]);
                                        mu2[index]= (epoch/(epoch+1.0))*mu2[index]+(curr_op[neuron]/(epoch+1));
                                } else {
                                        op3[index]+=(epoch/(epoch+1.0))*(curr_op[neuron]-mu3[index])*(pruning3[index]-mu3[index]);
                                        sig3[index]+=(epoch/(epoch+1.0))*(curr_op[neuron]-mu3[index])*(curr_op[neuron]-mu3[index]);
                                        mu3[index]= (epoch/(epoch+1.0))*mu3[index]+(curr_op[neuron]/(epoch+1));
                                }
				index-=neuron;
                        }
                        if(count==32768)
                                cudaMemcpy(pruning1+(indexx*this->num_*this->top_dim_)+(n*this->top_dim_), top_data + n * this->top_dim_,this->top_dim_,cudaMemcpyDeviceToHost);
                        else if(count==8192)
                                cudaMemcpy(pruning2+(indexx*this->num_*this->top_dim_)+(n*this->top_dim_), top_data + n * this->top_dim_,this->top_dim_,cudaMemcpyDeviceToHost);
                        else
                                cudaMemcpy(pruning3+(indexx*this->num_*this->top_dim_)+(n*this->top_dim_), top_data + n * this->top_dim_,this->top_dim_,cudaMemcpyDeviceToHost);
          }
      } else {
	  cudaMalloc((void **)&gaf,sizeof(float)*count);
	  if (norr){
		this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
	              top_data + n * this->top_dim_,NULL,true);
		cudaDeviceSynchronize();
		cudaMemcpy(curr_op,top_data + n * this->top_dim_,this->top_dim_,cudaMemcpyDeviceToHost);
		//update mean and var now.
		int index=indexx*this->num_*this->top_dim_+(n*this->top_dim_);
                        for (int neuron=0; neuron<count; neuron++){
                                index+=neuron;
                                if (count==32768){
                                        sig1[index]+=(epoch/(epoch+1.0))*(curr_op[neuron]-mu1[index])*(curr_op[neuron]-mu1[index]);
                                        op1[index]+=(epoch/(epoch+1.0))*(curr_op[neuron]-mu1[index])*(pruning1[index]-mu1[index]);
					if (sig1[index]==0)
                                                new_val[neuron]=1.0;
                                        else
                                                new_val[neuron]=abs(op1[index]/sig1[index]);
                                        mu1[index]= (epoch/(epoch+1.0))*mu1[index]+(curr_op[neuron]/(epoch+1));
                                } else if (count==8192){
                                        op2[index]+=(epoch/(epoch+1.0))*(curr_op[neuron]-mu2[index])*(pruning2[index]-mu2[index]);
                                        sig2[index]+=(epoch/(epoch+1.0))*(curr_op[neuron]-mu2[index])*(curr_op[neuron]-mu2[index]);
                                        if (sig2[index]==0)
                                                new_val[neuron]=1.0;
                                        else
                                                new_val[neuron]=abs(op2[index]/sig2[index]);
                                        mu2[index]= (epoch/(epoch+1.0))*mu2[index]+(curr_op[neuron]/(epoch+1));
                                } else {
                                        op3[index]+=(epoch/(epoch+1.0))*(curr_op[neuron]-mu3[index])*(pruning3[index]-mu3[index]);
                                        sig3[index]+=(epoch/(epoch+1.0))*(curr_op[neuron]-mu3[index])*(curr_op[neuron]-mu3[index]);
                                        if (sig3[index]==0)
                                                new_val[neuron]=1.0;
                                        else
                                                new_val[neuron]=abs(op3[index]/sig3[index]);
                                        mu3[index]= (epoch/(epoch+1.0))*mu3[index]+(curr_op[neuron]/(epoch+1));
                                }
				index-=neuron;
                        }
		cudaMemcpy(gaf,new_val,count,cudaMemcpyHostToDevice);
                        if (count == 32768){
                                Threshold_pruning<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, gaf, ruing1+(indexx*this->num_*this->top_dim_)+(n*this->top_dim_), top_data + n * this->top_dim_);
                        } else if (count==8192) {
                                Threshold_pruning<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, gaf, ruing2+(indexx*this->num_*this->top_dim_)+(n*this->top_dim_), top_data + n * this->top_dim_);
                        } else {
                                Threshold_pruning<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, gaf, ruing3+(indexx*this->num_*this->top_dim_)+(n*this->top_dim_), top_data + n * this->top_dim_);
                        }
                        if(count==32768)
                                cudaMemcpy(pruning1+(indexx*this->num_*this->top_dim_)+(n*this->top_dim_), top_data + n * this->top_dim_,this->top_dim_,cudaMemcpyDeviceToHost);
                        else if(count==8192)
                                cudaMemcpy(pruning2+(indexx*this->num_*this->top_dim_)+(n*this->top_dim_), top_data + n * this->top_dim_,this->top_dim_,cudaMemcpyDeviceToHost);
                        else
                                cudaMemcpy(pruning3+(indexx*this->num_*this->top_dim_)+(n*this->top_dim_), top_data + n * this->top_dim_,this->top_dim_,cudaMemcpyDeviceToHost);
		CUDA_POST_KERNEL_CHECK;
	  } else {
		if (count == 32768)
			this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight, top_data + n * this->top_dim_, ruing1+(indexx*this->num_*this->top_dim_)+(n*this->top_dim_), false);
		else if (count == 8192) 
			this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight, top_data + n * this->top_dim_, ruing2+(indexx*this->num_*this->top_dim_)+(n*this->top_dim_), false);
		else
			this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight, top_data + n * this->top_dim_, ruing3+(indexx*this->num_*this->top_dim_)+(n*this->top_dim_), false);
		cudaDeviceSynchronize();
		cudaMemcpy(curr_op,top_data + n * this->top_dim_,this->top_dim_,cudaMemcpyDeviceToHost);
		//update mean and var now.
			int index=indexx*this->num_*this->top_dim_+(n*this->top_dim_);
                        for (int neuron=0; neuron<count; neuron++){
                                index+=neuron;
                                if (count==32768){
                                        sig1[index]+=(epoch/(epoch+1.0))*(curr_op[neuron]-mu1[index])*(curr_op[neuron]-mu1[index]);
                                        op1[index]+=(epoch/(epoch+1.0))*(curr_op[neuron]-mu1[index])*(pruning1[index]-mu1[index]);
                                        if (sig1[index]==0)
                                                new_val[neuron]=1.0;
                                        else    
                                                new_val[neuron]=abs(op1[index]/sig1[index]);
                                        mu1[index]= (epoch/(epoch+1.0))*mu1[index]+(curr_op[neuron]/(epoch+1));
                                } else if (count==8192){
                                        op2[index]+=(epoch/(epoch+1.0))*(curr_op[neuron]-mu2[index])*(pruning2[index]-mu2[index]);
                                        sig2[index]+=(epoch/(epoch+1.0))*(curr_op[neuron]-mu2[index])*(curr_op[neuron]-mu2[index]);
                                        if (sig2[index]==0)
                                                new_val[neuron]=1.0;
                                        else    
                                                new_val[neuron]=abs(op2[index]/sig2[index]);
                                        mu2[index]= (epoch/(epoch+1.0))*mu2[index]+(curr_op[neuron]/(epoch+1));
                                } else {
                                        op3[index]+=(epoch/(epoch+1.0))*(curr_op[neuron]-mu3[index])*(pruning3[index]-mu3[index]);
                                        sig3[index]+=(epoch/(epoch+1.0))*(curr_op[neuron]-mu3[index])*(curr_op[neuron]-mu3[index]);
                                        if (sig3[index]==0)
                                                new_val[neuron]=1.0;
                                        else    
                                                new_val[neuron]=abs(op3[index]/sig3[index]);
                                        mu3[index]= (epoch/(epoch+1.0))*mu3[index]+(curr_op[neuron]/(epoch+1));
                                }
				index-=neuron;
                        }
                        if(count==32768)
                                cudaMemcpy(pruning1+(indexx*this->num_*this->top_dim_)+(n*this->top_dim_), top_data + n * this->top_dim_,this->top_dim_,cudaMemcpyDeviceToHost);
                        else if(count==8192)
                                cudaMemcpy(pruning2+(indexx*this->num_*this->top_dim_)+(n*this->top_dim_), top_data + n * this->top_dim_,this->top_dim_,cudaMemcpyDeviceToHost);
                        else    
                                cudaMemcpy(pruning3+(indexx*this->num_*this->top_dim_)+(n*this->top_dim_), top_data + n * this->top_dim_,this->top_dim_,cudaMemcpyDeviceToHost);
                CUDA_POST_KERNEL_CHECK;
	  }
      }
	free(curr_op);
	cudaFree(gaf);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
    if (this->phase_ == TRAIN){
	if(this->top_dim_ == 4096) //dimension of last layer
		images+=this->num_;
	if (images >=50000){ //MNIST has 60k images in total
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
