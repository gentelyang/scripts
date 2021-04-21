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
        self.types = [np.float16]
        # self.debug = True
        # enable check grad
        self.enable_backward = False

obj = TestSqrt(paddle.fluid.layers.sqrt)

def test_sqrt_base():
    """
    default
    """
    x = np.random.random([3072, 768]).astype(np.float16)
    res = np.sqrt(x)
    obj.run(res=res, x=x)