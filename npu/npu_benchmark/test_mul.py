#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
File: test_mul.py.py
Authors: liyang109
Date: 2021/3/23 3:53
"""
from apibase import APIBase
import paddle
import numpy as np


class TestMul(APIBase):
    """
    test mul
    """
    def hook(self):
        """
        implement
        """
        self.types = [np.float16]
        # self.debug = True
        # enable check grad
        self.enable_backward = False

obj = TestMul(paddle.fluid.layers.mul)

def test_shape_base():
    """
    base,x,y=float,forward has diff on xpu
    """
    x = np.random.random([8192, 3072]).astype(np.float16)
    y = np.random.random([3072, 768]).astype(np.float16)
#    res = np.matmul(x, y)
    res = [1]
    obj.run(res=res, x=x, y=y)