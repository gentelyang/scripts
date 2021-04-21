#!/usr/bin/env bash
fleetx_path=/workspace/FleetX
version=`date -d @$(git log -1 --pretty=format:%ct) "+%Y%m%d"`
fleet_cpu_model_list=(ctr_app w2v)
fleet_gpu_model_list=(resnet_app vgg_app bert_app transformer_app)
fleet_test_models=(resnet_single)


function setproxy(){
  export http_proxy=${proxy}
  export https_proxy=${proxy}
}


function unsetproxy(){
  unset http_proxy
  unset https_proxy
}


function kill_fleetx_process(){
  kill `ps -ef|grep python|awk '{print $2}'`
}


function check_result() {
    if [ $? -ne 0 ];then
      echo -e "\033[4;31;42m$1 model runs failed, please check your pull request or modify test case! \033[0m"
      exit 1
    else
      echo -e "\033[4;37;42m$1 model runs successfully, congratulations! \033[0m"
    fi
}


function before_hook() {
    wget --no-check-certificate  https://fleet.bj.bcebos.com/test/fleet_x-0.0.8-py2.py3-none-any.whl
    pip install fleet_x-0.0.8-py2.py3-none-any.whl
    pip install opencv-python==4.2.0.32
    echo "fleetx installed succ"

    wget https://paddle-wheel.bj.bcebos.com/develop-gpu-cuda9-cudnn7-openblas%2Fpaddlepaddle_gpu-2.1.0_dev0.post90-cp27-cp27mu-linux_x86_64.whl --no-check-certificate
    mv develop-gpu-cuda9-cudnn7-openblas%2Fpaddlepaddle_gpu-2.1.0_dev0.post90-cp27-cp27mu-linux_x86_64.whl paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl
    pip install paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl
    echo "paddlepaddle installed succ"
}


function resnet_single() {
    cd ${fleetx_path}/deprecated/fleet_x/examples/
    sed -i "s/(2)/(1)/g" resnet_single.py
    fleetrun --gpus 0 resnet_single.py
    check_result $FUNCNAME
    kill_fleetx_process
}


function resnet_app() {
    cd ${fleetx_path}/deprecated/fleet_x/examples/
    fleetrun --gpus 0,1 resnet_app.py
    check_result $FUNCNAME
    kill_fleetx_process
}


function vgg_app() {
    cd ${fleetx_path}/examples
    fleetrun --gpus 0,1 vgg_app.py
    check_result $FUNCNAME
    kill_fleetx_process
}



function bert_app() {
    cd ${fleetx_path}/examples
    fleetrun --gpus 0,1 bert_app.py
    check_result $FUNCNAME
    kill_fleetx_process
}


function transformer_app() {
    cd ${fleetx_path}/examples
    fleetrun --gpus 0,1 transformer_app.py
    check_result $FUNCNAME
    kill_fleetx_process
}


function ctr_app() {
    cd ${fleetx_path}/examples
    sed -i "s/epoch=10/epoch=1/g" ctr_app.py
    sed -i "s/train_data/raw_data/g" ctr_app.py
    mkdir raw_data
    cp -r /root/.cache/dist_data/serving/criteo_ctr_with_cube/raw_data/part-0 ./raw_data/
    cp -r /root/.cache/dist_data/serving/criteo_ctr_with_cube/raw_data/part-1 ./raw_data/
    fleetrun ctr_app.py
    check_result $FUNCNAME
    kill_fleetx_process
}


function w2v() {
    cd ${fleetx_path}/examples
    fleetrun word2vec_app.py
    check_result $FUNCNAME
    kill_fleetx_process
}


function run_cpu_models(){
      for model in ${fleet_cpu_model_list[@]}
      do
        echo "===========${model} run begin==========="
        $model
        sleep 3
        echo "===========${model} run  end ==========="
      done
}

function run_gpu_models(){
      for model in ${fleet_gpu_model_list[@]}
      do
        echo "===========${model} run begin==========="
        $model
        sleep 3
        echo "===========${model} run  end ==========="
      done
}

function run_test_models(){
      for model in ${fleet_test_models[@]}
      do
        echo "===========${model} run begin==========="
        $model
        sleep 3
        echo "===========${model} run  end ==========="
      done
}

function end_hook(){
  cd ${fleetx_path}/examples
  rm -rf *.data
  rm -rf *.tar.gz
  echo "===========files==========="
  ls -hlst
  echo "=========== end ==========="
}

main() {
    before_hook
    run_test_models
#    run_cpu_models
#    run_gpu_models
#    end_hook
}


main$@
