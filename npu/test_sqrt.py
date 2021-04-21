
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
File: test_sqrt.py
Authors: liyang109
Date: 2021/3/23 3:53
"""
from apibase import APIBase
import paddle
import numpy as np


class TestSqrt(APIBase):
    """
    test sqrt
    """
    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float16]
        # self.debug = True
        # enable check grad
        self.enable_backward = False

obj = TestSqrt(paddle.fluid.layers.sqrt)

def test_sqrt_base():
    """
    default
    """
    x = np.array([1, 2, 3])
    res = np.sqrt(x)
    obj.run(res=res, x=x)

def test_sqrt():
    """
    x = np.array([0.9, 0.8, 0.7, 0.6])
    """
    x = np.array([0.9, 0.8, 0.7, 0.6])
    res = np.sqrt(x)
    obj.run(res=res, x=x)

def test_sqrt1():
    """
    x_data_type=3-D tensor
    """
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    res = np.sqrt(x)
    obj.run(res=res, x=x)
