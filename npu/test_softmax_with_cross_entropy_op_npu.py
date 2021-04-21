#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
File: test_softmax_with_cross_entropy.py
Authors: liyang109
Date: 2021/3/23 3:53
"""
import paddle
import numpy as np
import paddle.fluid as fluid

if fluid.is_compiled_with_cuda() is True:
    places = [fluid.CPUPlace(), fluid.CUDAPlace(0)]
elif fluid.is_compiled_with_xpu() is True:
    places = [fluid.CPUPlace(), paddle.NPUPlace(7)]
else:
    places = [fluid.CPUPlace()]
types = [np.float32]

def test_softmax_with_cross_entropy_static1():
    """
    default
    """
    paddle.enable_static()
    for place in places:
        for t in types:
            x = np.arange(8).reshape(8).astype(t)
            label = np.array([[1]], dtype='int64')
            feed = {'x': x, 'label': label}
            main_program = fluid.Program()
            startup_program = fluid.Program()
            with fluid.program_guard(main_program=main_program, startup_program=startup_program):
                logits1 = paddle.static.data(name="x", shape=x.shape, dtype=t)
                label1 = paddle.static.data(name="label", shape=label.shape, dtype='int64')
                logits1.stop_gradient = False
                output = fluid.layers.softmax_with_cross_entropy(logits=logits1, label=label1)
                loss = paddle.mean(output)
                g = fluid.gradients(loss, logits1)
                exe = fluid.Executor(place)
                exe.run(startup_program)
                out, g = exe.run(main_program, feed=feed, fetch_list=[output, g])
                assert np.allclose(out, [[6.45833969]], atol=0.005, rtol=0.05, equal_nan=True)