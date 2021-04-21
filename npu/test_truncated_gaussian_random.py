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
  * @file test_truncated_gaussian_random.py
  * @author liyang109@baidu.com
  * @date 2021-03-24 15:11
  * @brief 
  *
  **************************************************************************/
"""
import paddle
import numpy as np
import paddle.fluid as fluid
# global params

if fluid.is_compiled_with_cuda() is True:
    places = [fluid.CPUPlace(), fluid.CUDAPlace(0)]
elif paddle.is_compiled_with_npu() is True:
    places = [fluid.CPUPlace(), paddle.NPUPlace(7)]
else:
    places = [fluid.CPUPlace()]

def test_static_truncated_gaussian_random1():
    """
    test_truncated_gaussian_random_static1
    """
    for place in places:
        paddle.enable_static()
        program = fluid.Program()
        block = program.global_block()
        vout = block.create_var(name="Out")
        outputs = ["Out"]
        attrs = {
            "shape": [10000],
            "mean": .0,
            "std": 1.,
            "seed": 10,
        }
        op = block.append_op(
            type="truncated_gaussian_random", outputs={"Out": vout}, attrs=attrs)
        op.desc.infer_var_type(block.desc)
        op.desc.infer_shape(block.desc)
        fetch_list = []
        for var_name in outputs:
            fetch_list.append(block.var(var_name))
        exe = fluid.Executor(place)
        outs = exe.run(program, fetch_list=fetch_list)
        tensor = outs[0]
        print("-----np.mean(tensor):", np.mean(tensor))
        assert np.allclose(np.mean(tensor), 0.016646797, atol=0.005, rtol=0.05, equal_nan=True)
        assert np.allclose(np.var(tensor), 0.782568, atol=0.005, rtol=0.05, equal_nan=True)