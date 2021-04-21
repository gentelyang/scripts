#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
# ======================================================================
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
# ======================================================================
"""
/***************************************************************************
  *
  * Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
  * @file test_mean.py
  * @author liyang109@baidu.com
  * @date 2021/3/23 3:53
  * @brief
  *
  **************************************************************************/
"""
from apibase import APIBase
import paddle
import numpy as np


class TestMean(APIBase):
    """
    test paddle.mean api
    """
    def hook(self):
        """
        implement
        """
        self.types = [np.float32]
        self.enable_backward = False

obj = TestMean(paddle.fluid.layers.mean)
# mean会报错
def test_mean1():
    """
    run axis is int and axis range[-2,2), axis<0
    :return:
    """
    x = np.array([[1, 2, 5, 4]]).astype('float32')
    res = np.mean(x)
    obj.run(res=res, x=x)

def test_mean2():
    """
    run axis is int and axis range[-2,2), axis>0
    :return:
    """
    x = np.array([[1, 2, 5, 4]]).astype('float32')
    res = np.mean(x)
    obj.run(res=res, x=x)

def test_mean3():
    """
    run axis is list, axis=[0,1], res should [1, 1, 3],
    :return:
    """
    x = np.array([[[2.0, 3.0, 4.0]], [[4.0, 3.0, 4.0]]]).astype('float32')
    res = np.mean(x, axis=(0, 1))
    obj.run(res=res, x=x, axis=[0, 1])

def test_mean4():
    """
    run axis is list, dim=3, axis=[0,1,2], res should [1, 1, 3],
    :return:
    """
    x = np.array([[[2.0, 3.0, 4.0]], [[4.0, 3.0, 4.0]]]).astype('float32')
    res = np.mean(x, axis=(0, 1, 2))
    obj.run(res=[res], x=x, axis=[0, 1, 2])

def test_mean5():
    """
    run axis is int and axis range[-2,2), axis<0, keepdim=False
    :return:
    """
    x = np.array([[1, 2, 5, 4]]).astype('float64')
    res = np.mean(x, axis=-2, keepdims=False)
    obj.run(res=res, x=x, axis=-2, keepdim=False)

def test_mean6():
    """
    run axis is int and axis range[-2,2), axis>0, keepdim=True
    :return:
    """
    x = np.array([[1, 2, 5, 4]]).astype('float64')
    res = np.mean(x, axis=1, keepdims=True)
    obj.run(res=res, x=x, axis=1, keepdim=True)

def test_mean7():
    """
    run axis is list, axis=[0,2], res should [1, 1, 3], keepdim=True
    static is ok, but dygraph has bug， waitting for fix.
    :return:
    """
    x = np.array([[[2.0, 3.0, 4.0]], [[4.0, 3.0, 4.0]]]).astype('float32')
    res = np.mean(x, axis=(0, 2), keepdims=True)
    obj.run(res=res, x=x, axis=[0, 2], keepdim=True)

def test_mean8():
    """
    run axis is list, dim=3, axis=[0,1,2], res should [1, 1, 3], keepdim=False
    static is ok, but dygraph has bug， waitting for fix.
    :return:
    """
    x = np.array([[[2.0, 3.0, 4.0]], [[4.0, 3.0, 4.0]]]).astype('float32')
    res = np.mean(x, axis=(0, 1, 2), keepdims=False)
    obj.run(res=[res], x=x, axis=[0, 1, 2], keepdim=False)

def test_mean9():
    """
    run axis is tuple, dim=3, axis=(0,2), res should [1, 1, 3], keepdim=True
    :return:
    """
    x = np.array([[[2.0, 3.0, 4.0]], [[4.0, 3.0, 4.0]]]).astype('float32')
    res = np.mean(x, axis=(0, 2), keepdims=True)
    obj.run(res=res, x=x, axis=(0, 2), keepdim=True)

def test_mean10():
    """
    run axis is tuple, dim=3, axis=(0,1,2), res should [1, 1, 3], keepdim=False
    :return:
    """
    x = np.array([[[2.0, 3.0, 4.0]], [[4.0, 3.0, 4.0]]]).astype('float32')
    res = np.mean(x, axis=(0, 1, 2), keepdims=False)
    obj.run(res=[res], x=x, axis=(0, 1, 2), keepdim=False)

def test_mean11():
    """
    run axis is tuple, dim=3, axis=(-3,1,2), res should [1, 1, 3], keepdim=False
    static is ok, but dygraph has bug， waitting for fix.
    :return:
    """
    x = np.array([[[2.0, 3.0, 4.0]], [[4.0, 3.0, 4.0]]]).astype('float32')
    res = np.mean(x, axis=(-3, 1, 2), keepdims=False)
    obj.run(res=[res], x=x, axis=(0, 1, 2), keepdim=False)