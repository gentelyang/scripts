#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
File: test_scale.py
Authors: liyang109
Date: 2021/3/23 3:53
"""
from apibase import APIBase
from apibase import randtool
import paddle
import numpy as np


class TestScale(APIBase):
    """
    test abs
    """
    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        self.enable_backward = False

obj = TestScale(paddle.fluid.layers.scale)


def test_scale_base():
    """
    base,bias_after_scale=True,act=None,has diff on xpu,cpu
    """
    x = np.array([[2.3, 3.9, 3.2], [1.3, 5.4, 3.8]])
    scale = 1
    bias = 0
    res = scale * np.array(x) + bias
    obj.base(res=res, x=x)