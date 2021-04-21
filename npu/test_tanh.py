#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
File: test_tanh.py
Authors: liyang109
Date: 2021/3/23 3:53
"""
from apibase import APIBase
from apibase import randtool
import paddle
import numpy as np


class TestTanh(APIBase):
    """
    test tanh
    """
    def hook(self):
        """
        implement
        """
        # np.float16 not support cpu
        self.types = [np.float32, np.float64]
        # self.debug = True
        # enable check grad
        self.enable_backward = False

obj = TestTanh(paddle.fluid.layers.tanh)

def test_tanh_base():
    """
    base
    """
    x = np.array([2, 3, 4])
    res = np.tanh(x)
    obj.base(res=res, x=x)

def test_tanh():
    """
    x=+
    """
    x = randtool('float', 1, 10, [3, 3, 3])
    res = np.tanh(x)
    obj.run(res=res, x=x)