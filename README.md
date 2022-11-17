# Deploy KoGPT with Triton Server

## Introduction
This repository is a template of NVIDIA Triton Inference Server. There are some requirements for deploying KoGPT with triton server and all pipelines are written in [build.sh](./build.sh) script. Follow the below sections and deploy a large language model.

## Getting started

Let's deploy KoGPT model with NVIDIA Triton Server and FasterTransformer library! They highly recommend to use docker container for working properly, so we will build some docker images including our custom server. The images are really large, so make sure that you have at least 100GB of free space.

### Build a custom triton server
Before the works, this guide is written for single GPU environment. Of course NVIDIA Triton server supports multi-GPUs, so if you are considering the multi-GPU environment, you have to modify some configurations. First, update the number of GPUs in `build.sh` script:
```bash
export NUM_INFERENCE_GPUS=8
```
And then the configuration file (`triton-model-store/gptj/fastertransformer/config.pbtxt`) of triton model should be modified:
```pbtxt
parameters {
  key: "model_checkpoint_path"
  value: {
    string_value: "/workspace/triton-model-store/gptj/fastertransformer/1/8-gpu"
  }
}
```
Because the weight conversion is performed according to the number of GPUs, the output directory will be changed from `1-gpu` to `8-gpu`. Hence you need to change the model path parameter.

So, if all prerequisites are done, then just simply run the below command:
```bash
$ bash build.sh
```

### Run the triton server
After you build the custom triton server, you can find `novelist_triton_server` docker image. It is based on the original triton image with fastertransformer backend library. The difference is that the converted KoGPT weights and the kernel auto-tuning profile are in the custom image. Therefore, all you need is to run a new triton server with the converted weights.

```bash
$ docker run --rm --gpus=all --shm-size=4G -p 8888:8888 novelist_triton_server \
    /opt/tritonserver/bin/tritonserver --model-repository=./triton-model-store/gptj/ --http-port 8888
```

You can change the port 8888 to your own one. You should change the port-forwarding of docker command either (i.e. `-p 8888:8888`).

## Manually build the image
Instead of using the composed build script, you may want to do step-by-step. In this section, you can stop and resume anytime. It can help you troubleshoot a problem occurred while running the script.

### Build FasterTransformer library
To convert weights and auto-tune kernels, we need FasterTransformer library. Unfortunately, there is no any pre-built library. So we will clone the repository and build the image. First, choose the container version:
```bash
$ export CONTAINER_VERSION=22.04
$ export FT_DOCKER_IMAGE=fastertransformer:${CONTAINER_VERSION}
```
Next, clone the repository and build with dockerfile:
```bash
$ git clone https://github.com/NVIDIA/FasterTransformer.git
$ docker build --rm --build-arg DOCKER_VERSION=${CONTAINER_VERSION} -t ${FT_DOCKER_IMAGE} -f docker/Dockerfile.torch .
```

### Build Triton Server with FasterTransformer backend
We also need a docker image of Triton Server with FasterTransformer backend. There is a pre-built Triton Server image, but we have to manually build for FasterTransformer backend to use faster language-model inference. Similar to the above section, choose the container version:
```bash
$ export CONTAINER_VERSION=22.04
$ export TRITON_DOCKER_IMAGE=triton_with_ft:${CONTAINER_VERSION}
```
Next, clone the repository and build with dockerfile:
```bash
$ git clone https://github.com/triton-inference-server/fastertransformer_backend.git
$ docker build --rm --build-arg TRITON_VERSION=${CONTAINER_VERSION} -t ${TRITON_DOCKER_IMAGE} -f docker/Dockerfile .
```

### Download huggingface KoGPT model
Now we are going to download KoGPT weights from huggingface hub. Make sure `git-lfs` is installed. If not, run the below command:
```bash
$ sudo apt install git-lfs
$ git lfs install
```
Instead of using python for downloading the weights, you can simply download them by using git:
```bash
$ git clone -b KoGPT6B-ryan1.5b-float16 --single-branch https://huggingface.co/kakaobrain/kogpt
```

### Convert the weights and auto-tune kernels
To use the model through FasterTransformer, the model should be converted to the corresponding format. In addition, to gain some performance enhancement, it is recommended to auto-tune some kernels to find the best algorithms. With using the docker image which we build in the above section, you can simply do by:
```bash
$ export NUM_INFERENCE_GPUS=1
$ docker run --rm --gpus=all -e NUM_INFERENCE_GPUS=${NUM_INFERENCE_GPUS} -v .:/ft_workspace ${FT_DOCKER_IMAGE} \
    /bin/bash -c \
        'python ./examples/pytorch/gptj/utils/huggingface_gptj_ckpt_convert.py \
            --ckpt-dir /ft_workspace/kogpt/ \
            --output-dir /ft_workspace/triton-model-store/gptj-weights/ \
            --n-inference-gpus ${NUM_INFERENCE_GPUS} && \
        ./build/bin/gpt_gemm 8 1 32 16 256 16384 50400 1 1 && \
        mv gemm_config.in /ft_workspace/'

```
Then you can see `gemm_config.in` file and `triton-model-store/gptj/fastertransformer/1/1-gpu` directory.

### Create a new custom Triton server
It's almost done! You can simply test with:
```bash
$ docker run --rm --gpus=all --shm-size=4G -v .:/ft_workspace -p 8888:8888 ${TRITON_DOCKER_IMAGE} \
    /opt/tritonserver/bin/tritonserver --model-repository=/ft_workspace/triton-model-store/gptj/ --http-port 8888
```
However, you have to manage both docker image and model weights. Let's copy the files into the server image. It is optional.
```bash
export SERVER_CONTAINER_NAME=novelist_triton_server
export SERVER_DOCKER_IMAGE=novelist_triton_server

docker create --name ${SERVER_CONTAINER_NAME} ${TRITON_DOCKER_IMAGE}
docker cp triton-model-store ${SERVER_CONTAINER_NAME}:/workspace/
docker cp gemm_config.in ${SERVER_CONTAINER_NAME}:/workspace/
docker commit ${SERVER_CONTAINER_NAME} ${SERVER_DOCKER_IMAGE}
docker rm ${SERVER_CONTAINER_NAME}
```
Now you have a new custom server image containing Triton Inference Server, FasterTransformer backend, and KoGPT model weights. You can upload it to hub and run without additional files.
