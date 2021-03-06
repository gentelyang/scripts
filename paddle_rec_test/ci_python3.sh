#!/bin/bash
export LD_LIBRARY_PATH=/opt/rh/devtoolset-2/root/usr/lib64:/opt/rh/devtoolset-2/root/usr/lib:/usr/local/lib64:/usr/local/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

function setproxy(){
  export http_proxy=${proxy}
  export https_proxy=${proxy}
}

function check_style() {
  set -e

  export PATH=/usr/bin:$PATH
  pre-commit install

  if ! pre-commit run -a; then
    git diff
    exit 1
  fi

  exit 0
}

function prepare(){
    #setproxy
    cd PaddleRec
    python -m pip uninstall paddle-rec -y
#     pip3 install skbuild 
#     pip3 install opencv-python==4.2.0.32 
    python setup.py install
    python -m pip uninstall paddlepaddle -y
    wget https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda10-cudnn7-mkl/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl --no-check-certificate
    python -m pip install paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl
    python -m pip install nose
    python -m pip install ruamel.yaml
    cd ../ && mkdir test_logs
    #   unset http_proxy, https_proxy
}

function run(){
    cases="test_paddlerec_features.py \
               test_paddlerec_features_new_config.py \
               test_paddlerec_mmoe.py \
               test_paddlerec_models.py \
               test_user_define.py"

    for file in ${cases}
    do
        echo ${file}
        #nosetests -s -v --with-html --html-report=test_logs/${file}.html ${file}
        nosetests -s -v ${file}
        rm -f *.yaml
        rm -rf increment* inference* logs
    done
}
prepare
