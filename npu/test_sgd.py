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
  * @file test_sgd.py
  * @author liyang109@baidu.com
  * @date 2021-03-24 15:46
  * @brief 
  *
  **************************************************************************/
"""
import paddle
import numpy as np
import paddle.fluid as fluid
# global params
types = [np.float64, np.float32]
if fluid.is_compiled_with_cuda() is True:
    places = [fluid.CPUPlace(), fluid.CUDAPlace(0)]
elif paddle.is_compiled_with_npu() is True:
    places = [fluid.CPUPlace(), paddle.NPUPlace(7)]
else:
    places = [fluid.CPUPlace()]



def test_static_learning_rate():
    """
    test_static_learning_rate
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
                sgd = paddle.optimizer.SGD(parameters=[bilinear.weight, bilinear.bias],
                                           learning_rate=0.01, weight_decay=0.0)
                output = bilinear(input1, input2)
                output = paddle.mean(output)
                sgd.minimize(output)
                exe = fluid.Executor(place)
                exe.run(startup_program)
                x = np.arange(3, 6).reshape((3, 1)).astype(t)
                y = np.arange(6, 12).reshape((3, 2)).astype(t)
                for i in range(10):
                    out, weight = exe.run(main_program, feed={'x': x, 'y': y}, fetch_list=[output, bilinear.weight])
                res = np.array([[[1.1666663, 1.0666664]], [[1.1666663, 1.0666664]],
                                [[1.1666663, 1.0666664]], [[1.1666663, 1.0666664]]])
                assert np.allclose(np.array(weight).shape, res.shape)