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
  * @file test_assign.py
  * @author liyang109@baidu.com
  * @date 2021-03-24 14:56
  * @brief 
  *
  **************************************************************************/
"""
import paddle
import paddle.fluid as fluid
import numpy as np
from apibase import compare

if fluid.is_compiled_with_cuda() is True:
    places = [fluid.CPUPlace(), fluid.CUDAPlace(0)]
elif paddle.is_compiled_with_npu() is True:
    places = [fluid.CPUPlace(), paddle.NPUPlace(7)]
else:
    places = [fluid.CPUPlace()]
types1 = [np.float32]
seed = 33


def test_assign_static1():
    """
    default,input=tensor,type=int,bool
    """
    paddle.enable_static()
    for t in types1:
        np.random.seed(seed)
        main_program = fluid.Program()
        startup_program = fluid.Program()
        input = np.arange(6).reshape([6]).astype(t)
        feed = {"input": input}
        with fluid.unique_name.guard():
            with fluid.program_guard(main_program=main_program, startup_program=startup_program):
                input1 = paddle.static.data(name="input", shape=input.shape, dtype=t)
                input1.stop_gradient = False
                output = paddle.fluid.layers.assign(input1)
                loss = paddle.mean(output)
                g = fluid.gradients(loss, input1)
                exe = fluid.Executor()
                exe.run(startup_program)
                out, g = exe.run(main_program, feed=feed, fetch_list=[output, g])
                compare(out, input)


