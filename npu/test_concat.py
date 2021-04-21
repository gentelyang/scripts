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
  * @file test_concat.py
  * @author liyang109@baidu.com
  * @date 2021-03-24 16:45
  * @brief 
  *
  **************************************************************************/
"""
import paddle
import numpy as np
import paddle.fluid as fluid
# global params
# np.float32 back Segmentation fault
types = [np.float32]

if fluid.is_compiled_with_cuda() is True:
    places = [fluid.CPUPlace(), fluid.CUDAPlace(0)]
elif paddle.is_compiled_with_npu() is True:
    places = [fluid.CPUPlace(), paddle.NPUPlace(7)]
else:
    places = [fluid.CPUPlace()]


def static_base(axis):
    """
    static_base, back error on npu
    """
    for place in places:
        for t in types:
            paddle.enable_static()
            main_program = fluid.Program()
            startup_program = fluid.Program()
            with fluid.program_guard(main_program=main_program, startup_program=startup_program):
                input1 = paddle.static.data(name="x", shape=[2, 3], dtype=t)
                input2 = paddle.static.data(name="y", shape=[2, 3], dtype=t)
                # input1.stop_gradient = False
                output = paddle.concat(x=(input1, input2), axis=axis)
                # g = fluid.gradients(output, input1)
                exe = fluid.Executor(place)
                exe.run(startup_program)
                x = np.arange(6).reshape((2, 3)).astype(t)
                y = np.arange(6).reshape((2, 3)).astype(t)
                out = exe.run(main_program, feed={'x': x, 'y': y}, fetch_list=[output])
    return out

def test_static_0():
    """
    static
    """
    out = static_base(0)
    res_out = np.array([[0., 1., 2.],
                        [3., 4., 5.],
                        [0., 1., 2.],
                        [3., 4., 5.]])
    assert np.allclose(out, res_out)