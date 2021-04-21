#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
function prepare(){
    env
    ln -s ../dist_fleet/thirdparty ./
    ln -s /ssd3/ly/cts_ce/dataset/data/ ./
    export LD_LIBRARY_PATH=/opt/_internal/cpython-3.7.0/lib/:/home/work/418.39/lib64/:${LD_LIBRARY_PATH}
    export PATH=/opt/_internal/cpython-3.7.0/bin/:${PATH}
    pip uninstall paddlepaddle-gpu -y
    pip uninstall paddlepaddle -y
    pip install nose
    pip install opencv-python==4.2.0.32
    pip install nose-html-reporting
    pip install ${IMAGE_NAME}
    echo "paddlepaddle install succ"
    unset http_proxy
    unset https_proxy
    echo "IMAGE_NAME is: ${IMAGE_NAME}"
}

function run(){
    cases="test_dist_fleet_static_vgg.py \
           test_dist_fleet_dygraph_api.py \
           test_dist_fleet_static_strategy.py \
           test_dist_fleet_static_launch.py \
           test_dist_fleet_static_fleetrun.py \
           test_dist_fleet_dygraph_gloo.py \
           test_dist_fleet_communicator_api.py \
           test_dist_fleet_init.py \
           test_dist_fleet_paddlecloudrolemaker.py \
           test_dist_fleet_userdefinedrolemaker.py \
           test_dist_fleet_utils_cloud_client.py \
           test_static_dygraph_look_ahead.py \
           test_dist_fleet_worker.py \
           test_dist_fleet_static_ctr.py \
           test_dist_fleet_ps_communicator_api.py \
           test_dist_fleet_dygraph_loss_consistent.py \
           "
    for file in ${cases}
    do
        echo ${file}
        nosetests -s -v  ${file}
    done
}

prepare
run
