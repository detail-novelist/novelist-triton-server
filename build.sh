export WORKSPACE=$(pwd)
export NUM_INFERENCE_GPUS=1

export CONTAINER_VERSION=22.04
export FT_DOCKER_IMAGE=fastertransformer:${CONTAINER_VERSION}
export TRITON_DOCKER_IMAGE=triton_with_ft:${CONTAINER_VERSION}

export SERVER_CONTAINER_NAME=novelist_triton_server
export SERVER_DOCKER_IMAGE=novelist_triton_server

# Build docker images of fastertransformer and its triton-server. 
cd ${WORKSPACE}
git clone https://github.com/NVIDIA/FasterTransformer.git
git clone https://github.com/triton-inference-server/fastertransformer_backend.git

cd FasterTransformer
docker build --rm --build-arg DOCKER_VERSION=${CONTAINER_VERSION} -t ${FT_DOCKER_IMAGE} -f docker/Dockerfile.torch .
cd ${WORKSPACE}

cd fastertransformer_backend
docker build --rm --build-arg TRITON_VERSION=${CONTAINER_VERSION} -t ${TRITON_DOCKER_IMAGE} -f docker/Dockerfile .
cd ${WORKSPACE}

# Download huggingface KoGPT weights and convert to fastertransformer format with kernel auto-tuning.
git clone -b KoGPT6B-ryan1.5b-float16 --single-branch https://huggingface.co/kakaobrain/kogpt
docker run --rm --gpus=all -e NUM_INFERENCE_GPUS=${NUM_INFERENCE_GPUS} -v ${WORKSPACE}:/ft_workspace ${FT_DOCKER_IMAGE} /bin/bash -c '
    python ./examples/pytorch/gptj/utils/huggingface_gptj_ckpt_convert.py \
        --ckpt-dir /ft_workspace/kogpt/ --output-dir /ft_workspace/triton-model-store/gptj-weights/ --n-inference-gpus ${NUM_INFERENCE_GPUS} && \
    ./build/bin/gpt_gemm 8 1 32 16 256 16384 50400 1 1 && \
    mv gemm_config.in /ft_workspace/'

# Remove unnecessary directories.
rm -rf fastertransformer_backend FasterTransformer kogpt

# Create a new triton server container and copy some requirements for composing a KoGPT inference server image.
docker create --name ${SERVER_CONTAINER_NAME} ${TRITON_DOCKER_IMAGE}
docker cp triton-model-store ${SERVER_CONTAINER_NAME}:/workspace/
docker cp gemm_config.in ${SERVER_CONTAINER_NAME}:/workspace/
docker commit ${SERVER_CONTAINER_NAME} ${SERVER_DOCKER_IMAGE}
docker rm ${SERVER_CONTAINER_NAME}

# Run the merged inference server image.
docker run --rm --gpus=all --shm-size=4G -p 8888:8888 novelist_triton_server /opt/tritonserver/bin/tritonserver --model-repository=./triton-model-store/gptj/ --http-port 8888