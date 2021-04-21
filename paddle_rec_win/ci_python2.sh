#!/bin/bash

function setproxy(){
  export http_proxy=${proxy}
  export https_proxy=${proxy}
}

function prepare(){
    setproxy
    cd PaddleRec
    pip uninstall paddle-rec -y
    python setup.py install
    pip uninstall paddlepaddle -y
    wget https://paddle-wheel.bj.bcebos.com/1.8.3-gpu-cuda10-cudnn7-mkl%2Fpaddlepaddle_gpu-1.8.3.post107-cp27-cp27mu-linux_x86_64.whl
    mv 1.8.3-gpu-cuda10-cudnn7-mkl%2Fpaddlepaddle_gpu-1.8.3.post107-cp27-cp27mu-linux_x86_64.whl paddlepaddle_gpu-1.8.3.post107-cp27-cp27mu-linux_x86_64.whl
    pip install paddlepaddle_gpu-1.8.3.post107-cp27-cp27mu-linux_x86_64.whl
    pip install nose
    pip install ruamel.yaml
    cd ../ && mkdir test_logs
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
#run
