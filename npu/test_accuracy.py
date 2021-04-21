#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
*
* Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
* @file test_accuracy.py
* @author liyang109
* @date 2021/3/23 3:53
* @brief test_accuracy
*
**************************************************************************/
"""
import paddle.fluid as fluid
import numpy as np
import paddle
if fluid.is_compiled_with_cuda() is True:
    devices = ['gpu:0', 'cpu']
elif paddle.is_compiled_with_npu() is True:
    devices = ['npu:0', 'cpu']
else:
    devices = ['cpu']

if fluid.is_compiled_with_cuda() is True:
    places = [fluid.CPUPlace(), fluid.CUDAPlace(0)]
elif paddle.is_compiled_with_npu() is True:
    places = [fluid.CPUPlace(), paddle.NPUPlace(7)]
else:
    places = [fluid.CPUPlace()]
value_types = [np.float32]

#目前npu不支持set_device
def test_Accuracy_static():
    """
    update,reset,value=array
    """
    paddle.enable_static()
    # for device in devices:
    for t in value_types:
        # paddle.set_device(device)
        batch_size = 128.0
        batch1_acc = np.array([0.88]).astype(t)
        accuracy_manager = fluid.metrics.Accuracy()
        accuracy_manager.update(value=batch1_acc, weight=batch_size)
        assert np.allclose(accuracy_manager.eval(), batch1_acc, atol=0.005, rtol=0.05, equal_nan=True)
