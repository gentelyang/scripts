#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
File: test_stack.py
Authors: liyang109
Date: 2021/3/23 3:53
"""
import paddle
import paddle.fluid as fluid
import numpy as np


def test_stack():
    """
    list tensor, run with static.
    :return:
    """
    paddle.enable_static()
    if fluid.is_compiled_with_cuda() is True:
        places = [fluid.CPUPlace(), fluid.CUDAPlace(0)]
    elif fluid.is_compiled_with_xpu() is True:
        places = [fluid.CPUPlace(), paddle.NPUPlace(7)]
    else:
        places = [fluid.CPUPlace()]
    x = fluid.data(name="x", shape=[1, 2], dtype=np.float64)
    y = fluid.data(name="y", shape=[1, 2], dtype=np.float64)
    res = paddle.stack([x, y], axis=0)
    for place in places:
        exe = fluid.Executor(place)
        x1 = np.array([[2, 1]]).astype(np.float64)
        y1 = np.array([[5, 6]]).astype(np.float64)
        res_val = exe.run(fluid.default_main_program(), feed={'x': x1, 'y': y1}, fetch_list=[res])
        expect = np.stack([x1, y1], axis=0)
        result = np.allclose(res_val, expect, atol=1e-6, rtol=1e-6, equal_nan=True)
        assert result
        assert res_val[0].shape == np.array(expect).shape