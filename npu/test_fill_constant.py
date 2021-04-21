#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:fill_constanttab:ft=python
"""
Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
File: test_fill_constant.py
Authors: liyang109
Date: 22021/3/23 3:53
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestExpand(APIBase):
    """
    test fill_constant
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

obj = TestExpand(paddle.fluid.layers.fill_constant)


def test_fill_constant1():
    """
    shape = (2, 3),x_type=np.int32
    """
    shape = (4,)
    dtype = np.float64
    value = 123
    res = np.full(shape=shape, fill_value=value, dtype=dtype)
    obj.run(res=res, shape=shape, value=value, dtype=dtype, force_cpu=True)