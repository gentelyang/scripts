#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
*
* Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
* @file test_functional_gelu.py
* @author liyang109
* @date 2021/3/23 3:53
* @brief test_functional_gelu
*
**************************************************************************/
"""
from apibase import APIBase
from apibase import randtool
from apibase import tanh
import paddle
import numpy as np
import math


class TestFunctionalGelu(APIBase):
    """
    test
    """
    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False

obj = TestFunctionalGelu(paddle.fluid.layers.gelu)

#基于npu的gelu精度有问题
def test_gelu_base():
    """
    base
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    # 算法
    arr = []
    for i in range(len(x.flatten())):
        arr.append(math.erf(x.flatten()[i] / math.sqrt(2)))
    arr = np.array(arr).reshape(x.shape)
    res = 0.5 * x * (1 + arr)
    obj.run(res=res, x=x)

def test_gelu():
    """
    default approximate=False
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    # 算法
    arr = []
    for i in range(len(x.flatten())):
        arr.append(math.erf(x.flatten()[i] / math.sqrt(2)))
    arr = np.array(arr).reshape(x.shape)
    res = 0.5 * x * (1 + arr)
    obj.run(res=res, x=x)

def test_gelu1():
    """
    approximate=True
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    # 算法
    res = 0.5 * x * (1 + tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * pow(x, 3))))
    obj.run(res=res, x=x)