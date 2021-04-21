#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
File: test_expand.py
Authors: liyang109
Date: 2021/3/23 3:53
"""
from apibase import APIBase
from apibase import randtool
import paddle.fluid as fluid
import pytest
import numpy as np
import paddle

#目前存在问题
class TestExpand(APIBase):
    """
    test expand
    """
    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float16]
        # self.debug = True
        # enable check grad
        # self.no_grad_var = {"x", "shape"}
        self.enable_backward = False

obj = TestExpand(paddle.fluid.layers.expand)


def test_expand1():
    """
    shape = (2, 3),x_type=np.int32
    """
    x = fluid.layers.fill_constant(shape=[12, 14], dtype="int32", value=3)
    expand_times = fluid.layers.fill_constant(shape=[2], dtype="int32", value=4)
    res = np.expand_dims(x, expand_times)
    obj.run(res=res, x=x, expand_times=expand_times)
