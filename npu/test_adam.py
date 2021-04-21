#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
#======================================================================
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
#======================================================================
"""
/***************************************************************************
  *
  * Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
  * @file test_adam.py
  * @author liyang109@baidu.com
  * @date 2021-03-24 16:07
  * @brief 
  *
  **************************************************************************/
"""
import paddle
import numpy as np
import paddle.fluid as fluid
# global params
types = [np.float32]

if fluid.is_compiled_with_cuda() is True:
    places = [fluid.CPUPlace(), fluid.CUDAPlace(0)]
elif paddle.is_compiled_with_npu() is True:
    places = [fluid.CPUPlace(), paddle.NPUPlace(7)]
else:
    places = [fluid.CPUPlace()]


def test_static_adam():
    """
    test_adam
    """
    for place in places:
        for t in types:
            paddle.enable_static()
            paddle.set_default_dtype(t)
            main_program = fluid.Program()
            startup_program = fluid.Program()
            with fluid.program_guard(main_program=main_program, startup_program=startup_program):
                input1 = paddle.static.data(name="x", shape=[3, 1])
                input2 = paddle.static.data(name="y", shape=[3, 2])
                bilinear = paddle.nn.Bilinear(1, 2, 4,
                                              weight_attr=fluid.initializer.ConstantInitializer(value=2.0),
                                              bias_attr=None)
                adam = paddle.optimizer.Adam(parameters=[bilinear.weight, bilinear.bias],
                                             learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-08)
                output = bilinear(input1, input2)
                output = paddle.mean(output)
                adam.minimize(output)
                exe = fluid.Executor(place)
                exe.run(startup_program)
                x = np.arange(3, 6).reshape((3, 1)).astype(t)
                y = np.arange(6, 12).reshape((3, 2)).astype(t)
                for i in range(10):
                    out, weight = exe.run(main_program, feed={'x': x, 'y': y}, fetch_list=[output, bilinear.weight])
                res = np.array([[[1.9000002, 1.9000002]], [[1.9000002, 1.9000002]],
                                [[1.9000002, 1.9000002]], [[1.9000002, 1.9000002]]])
                assert np.allclose(np.array(weight).shape, res.shape)