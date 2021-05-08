#!/usr/bin/env bash
export LD_LIBRARY_PATH=/opt/_internal/cpython-3.7.0/lib/:${LD_LIBRARY_PATH}
export PATH=/opt/_internal/cpython-3.7.0/bin/:${PATH}
fleetx_path=/workspace/FleetX
version=`date -d @$(git log -1 --pretty=format:%ct) "+%Y%m%d"`
fleet_cpu_model_list=(ctr_app w2v)
fleet_gpu_model_list=(resnet_static resnet_dygraph resnet_static_gradient_merge resnet_static_lamb \
resnet_static_lars resnet_static_recompute resnet_static_sharding resnet_static_amp resnet_static_communication \
resnet_static_communication_others resnet_static_communication_topolopy resnet_static_dgc resnet_static_localsgd \
resnet_static_op_fusion resnet_static_op_others resnet_static_op_overlap)
unset CUDA_VISIBLE_DEVICES

function setproxy(){
  export http_proxy=${proxy}
  export https_proxy=${proxy}
}


function unsetproxy(){
  unset http_proxy
  unset https_proxy
}


function kill_fleetx_process(){
  kill `ps -ef|grep resnet|awk '{print $2}'`
}


function check_result() {
    if [ $? -ne 0 ];then
      echo -e "\033[4;31;42m$1 model runs failed, please check your pull request or modify test case! \033[0m"
      #exit 1
    else
      echo -e "\033[4;37;42m$1 model runs successfully, congratulations! \033[0m"
    fi
}


function before_hook() { 
#    wget --no-check-certificate  https://fleet.bj.bcebos.com/test/fleet_x-0.0.8-py2.py3-none-any.whl
#    pip install fleet_x-0.0.8-py2.py3-none-any.whl
#    echo "fleetx installed succ"
    python -m pip install opencv-python==4.2.0.32
    python -m pip install pip==20.2.4
    python -m pip install scipy
    unsetproxy
    python -m pip install paddlepaddle-gpu==2.1.0.dev0.post110 -f https://paddlepaddle.org.cn/whl/cu110/mkl/develop.html
    echo "paddlepaddle installed succ"
}


function resnet_dygraph() {
    unsetproxy
    cd ${fleetx_path}/examples/resnet
    sed -i "s/epoch = 10/epoch = 1/g" train_fleet_dygraph.py
    fleetrun --gpus 0,1 train_fleet_dygraph.py
    check_result $FUNCNAME
}


function resnet_static() {
    setproxy
    cd ${fleetx_path}/examples/resnet
    sed -i "s/epoch = 10/epoch = 1/g" train_fleet_static.py
    fleetrun --gpus 0,1 train_fleet_static.py
    check_result $FUNCNAME
}

function resnet_static_gradient_merge() {
    setproxy
    cd ${fleetx_path}/examples/resnet
    sed -i "s/epoch = 10/epoch = 1/g" train_fleet_gradient_merge.py
    fleetrun --gpus 0,1 train_fleet_gradient_merge.py
    check_result $FUNCNAME
}

function resnet_static_lamb() {
    setproxy
    cd ${fleetx_path}/examples/resnet
    sed -i "s/epoch = 10/epoch = 1/g" train_fleet_lamb.py
    fleetrun --gpus 0,1 train_fleet_lamb.py
    check_result $FUNCNAME
}

function resnet_static_lars() {
    setproxy
    cd ${fleetx_path}/examples/resnet
    sed -i "s/epoch = 10/epoch = 1/g" train_fleet_lars.py
    fleetrun --gpus 0,1 train_fleet_lars.py
    check_result $FUNCNAME
}

function resnet_static_recompute() {
    setproxy
    cd ${fleetx_path}/examples/resnet
    sed -i "s/epoch = 10/epoch = 1/g" train_fleet_recompute.py
    fleetrun --gpus 0,1 train_fleet_recompute.py
    check_result $FUNCNAME
}

function resnet_static_sharding() {
    setproxy
    cd ${fleetx_path}/examples/resnet
    sed -i "s/epoch = 10/epoch = 1/g" train_fleet_sharding.py
    fleetrun --gpus 0,1 train_fleet_sharding.py
    check_result $FUNCNAME
}

function resnet_static_amp() {
    setproxy
    cd ${fleetx_path}/examples/resnet
    sed -i "s/epoch = 10/epoch = 1/g" train_fleet_static_amp.py
    fleetrun --gpus 0,1 train_fleet_static_amp.py
    check_result $FUNCNAME
}

function resnet_static_communication() {
    setproxy
    cd ${fleetx_path}/examples/resnet
    sed -i "s/epoch = 10/epoch = 1/g" train_fleet_static_communication_frequency.py
    fleetrun --gpus 0,1 train_fleet_static_communication_frequency.py
    check_result $FUNCNAME
}

function resnet_static_communication_others() {
    setproxy
    cd ${fleetx_path}/examples/resnet
    sed -i "s/epoch = 10/epoch = 1/g" train_fleet_static_communication_others.py
    fleetrun --gpus 0,1 train_fleet_static_communication_others.py
    check_result $FUNCNAME
}

function resnet_static_communication_topolopy() {
    setproxy
    cd ${fleetx_path}/examples/resnet
    sed -i "s/epoch = 10/epoch = 1/g" train_fleet_static_communication_topolopy.py
    fleetrun --gpus 0,1 train_fleet_static_communication_topolopy.py
    check_result $FUNCNAME
}

function resnet_static_dgc() {
    setproxy
    cd ${fleetx_path}/examples/resnet
    sed -i "s/epoch = 10/epoch = 1/g" train_fleet_static_dgc.py
    fleetrun --gpus 0,1 train_fleet_static_dgc.py
    check_result $FUNCNAME
}

function resnet_static_localsgd() {
    setproxy
    cd ${fleetx_path}/examples/resnet
    sed -i "s/epoch = 10/epoch = 1/g" train_fleet_static_localsgd.py
    fleetrun --gpus 0,1 train_fleet_static_localsgd.py
    check_result $FUNCNAME
}

function resnet_static_op_fusion() {
    setproxy
    cd ${fleetx_path}/examples/resnet
    sed -i "s/epoch = 10/epoch = 1/g" train_fleet_static_op_fusion.py
    fleetrun --gpus 0,1 train_fleet_static_op_fusion.py
    check_result $FUNCNAME
}

function resnet_static_op_others() {
    setproxy
    cd ${fleetx_path}/examples/resnet
    sed -i "s/epoch = 10/epoch = 1/g" train_fleet_static_others.py
    fleetrun --gpus 0,1 train_fleet_static_others.py
    check_result $FUNCNAME
}

function resnet_static_op_overlap() {
    setproxy
    cd ${fleetx_path}/examples/resnet
    sed -i "s/epoch = 10/epoch = 1/g" train_fleet_static_overlap.py
    fleetrun --gpus 0,1 train_fleet_static_overlap.py
    check_result $FUNCNAME
}
function widedeep {
    cd ${fleetx_path}/examples/wide_and_deep
    fleetrun --server_num=1 --worker_num=2 train.py
    check_result $FUNCNAME
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


main() {
    before_hook
    #run_cpu_models
    run_gpu_models
    
}


main$@
