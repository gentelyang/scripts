#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
*
* Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
* @file test_less_than.py
* @author liyang109
* @date 2021/3/23 3:53
* @brief test_less_than
*
**************************************************************************/
"""
from apibase import APIBase
from apibase import randtool
import paddle
import numpy as np


class TestEqual(APIBase):
    """
    test
    """
    def hook(self):
        """
        implement
        """
        self.types = [np.float32]
        self.debug = False
        # self.static = True
        # enable check grad
        self.enable_backward = False

obj = TestEqual(paddle.fluid.layers.less_than)

def test_less_than_dicimal():
    """
    less_than_dicimal
    """
    x_data = np.array([[2.0, 1.0, -3.5], [-2.7, 1.5, 3], [0, 4.1, 8.6]])
    y_data = np.array([[-2.0, 1.1, -3.5], [-2.5, 1.5, 3.5], [0.5, 4.2, 8.3]])
    res = np.less(x_data, y_data)
    obj.run(res=res, x=x_data, y=y_data)

def test_less_than_1D_tensor():
    """
    1D_tensor
    """
    x_data = np.array([1]).astype(np.float32)
    y_data = np.array([1, -1, 2, -4]).astype(np.float32)
    res = np.less(x_data, y_data)
    obj.run(res=res, x=x_data, y=y_data)

def test_less_than_broadcast_1():
    """
    broadcast_1
    """
    x_data = np.arange(1, 7).reshape((1, 2, 1, 3)).astype(np.float32)
    y_data = np.arange(0, 6).reshape((1, 2, 3)).astype(np.float32)
    res = np.less(x_data, y_data)
    obj.run(res=res, x=x_data, y=y_data)

def test_less_than_broadcast_2():
    """
    broadcast_2
    """
    x_data = np.arange(1, 3).reshape((1, 2)).astype(np.float32)
    y_data = np.arange(0, 4).reshape((2, 2)).astype(np.float32)
    res = np.less(x_data, y_data)
    obj.run(res=res, x=x_data, y=y_data)