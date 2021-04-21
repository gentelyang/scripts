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
        self.types = [np.float32]
        # self.debug = True
        # enable check grad
        self.enable_backward = False

obj = TestMul(paddle.fluid.layers.mul)

def test_shape_base():
    """
    base,x,y=float,forward has diff on xpu
    """
    x = np.arange(6).reshape((3, 2))
    y = np.arange(6).reshape((2, 3))
    res = np.matmul(x, y)
    obj.run(res=res, x=x, y=y)