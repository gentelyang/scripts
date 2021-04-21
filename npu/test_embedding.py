#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author:gentelyang  time:2021-03-23

#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
*
* Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
* @file test_Embedding.py
* @author liyang109
* @date 2021/3/23 3:53
*
**************************************************************************/
"""
import paddle
import numpy as np
import paddle.fluid as fluid

types = [np.int64]

if fluid.is_compiled_with_cuda() is True:
    places = [fluid.CPUPlace(), fluid.CUDAPlace(0)]
elif paddle.is_compiled_with_npu() is True:
    places = [fluid.CPUPlace(), paddle.NPUPlace(7)]
else:
    places = [fluid.CPUPlace()]


def test_static():
    """
    test_static
    """
    for place in places:
        for t in types:
            paddle.enable_static()
            main_program = fluid.Program()
            startup_program = fluid.Program()
            with fluid.program_guard(main_program=main_program, startup_program=startup_program):
                input1 = paddle.static.data(name="x", shape=[2, 4], dtype=t)
                embedding = paddle.nn.Embedding(10, 3,
                                                weight_attr=fluid.initializer.ConstantInitializer(value=1.0))
                sgd = paddle.optimizer.SGD(parameters=[embedding.weight],
                                           learning_rate=0.01, weight_decay=0.0)
                output = embedding(input1)
                output = paddle.mean(output)
                sgd.minimize(output)
                exe = fluid.Executor(place)
                exe.run(startup_program)
                x = np.array([[7, 2, 4, 5], [4, 3, 2, 9]], dtype=t)
                for i in range(10):
                    out, weight = exe.run(main_program, feed={'x': x}, fetch_list=[output, embedding.weight])
                res_weight = np.array([[1., 1., 1.],
                                       [1., 1., 1.],
                                       [0.9000004, 0.9000004, 0.9000004],
                                       [0.90000063, 0.90000063, 0.90000063],
                                       [0.9000004, 0.9000004, 0.9000004],
                                       [0.90000063, 0.90000063, 0.90000063],
                                       [1., 1., 1.],
                                       [0.90000063, 0.90000063, 0.90000063],
                                       [1., 1., 1.],
                                       [0.90000063, 0.90000063, 0.90000063]])
                res_out = np.array([[0.91000056]]),
                assert np.allclose(weight.shape, res_weight.shape)