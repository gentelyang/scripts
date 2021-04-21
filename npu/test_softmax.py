#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
*
* Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
* @file test_softmax.py
* @author liyang109
* @date 2021/3/23 3:53
* @brief test_softmax
*
**************************************************************************/
"""
from apibase import APIBase
from apibase import randtool
import paddle
import numpy as np
class TestSoftmax(APIBase):
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

obj = TestSoftmax(paddle.fluid.layers.softmax)

#npu上精度存在问题，cpu上ok；
def test_softmax_base():
    """
    base,has diff on xpu
    """
    x = np.array([[[2.0, 3.0, 4.0, 5.0],
                       [3.0, 4.0, 5.0, 6.0],
                       [7.0, 8.0, 8.0, 9.0]],
                      [[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [6.0, 7.0, 8.0, 9.0]]])
    res = np.array([[[0.0320586, 0.08714432, 0.23688282, 0.64391426],
                    [0.0320586, 0.08714432, 0.23688282, 0.64391426],
                    [0.07232949, 0.19661193, 0.19661193, 0.53444665]],
                    [[0.0320586, 0.08714432, 0.23688282, 0.64391426],
                    [0.0320586, 0.08714432, 0.23688282, 0.64391426],
                    [0.0320586, 0.08714432, 0.23688282, 0.64391426]]])
    obj.run(res=res, input=x)

def test_softmax():
    """
    default
    """
    x = np.array([[[2.0, 3.0, 4.0, 5.0],
                       [3.0, 4.0, 5.0, 6.0],
                       [7.0, 8.0, 8.0, 9.0]],
                      [[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [6.0, 7.0, 8.0, 9.0]]])
    res = np.array([[[0.0320586, 0.08714432, 0.23688282, 0.64391426],
                    [0.0320586, 0.08714432, 0.23688282, 0.64391426],
                    [0.07232949, 0.19661193, 0.19661193, 0.53444665]],
                    [[0.0320586, 0.08714432, 0.23688282, 0.64391426],
                    [0.0320586, 0.08714432, 0.23688282, 0.64391426],
                    [0.0320586, 0.08714432, 0.23688282, 0.64391426]]])
    obj.run(res=res, input=x)