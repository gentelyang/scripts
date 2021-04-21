#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
*
* Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
* @file test_equal.py
* @author liyang109
* @date 2021/3/23 3:53
* @brief test_equal
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

obj = TestEqual(paddle.fluid.layers.equal)

def test_equal():
    """
    default
    """
    x = randtool("int", -10, 10, [3, 3, 3])
    y = x
    res = np.equal(x, y)
    obj.run(res=res, x=x, y=y)

def test_equal1():
    """
    test broadcast
    """
    x = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    y = np.array([1, 2, 3])
    res = np.equal(x, y)
    obj.run(res=res, x=x, y=y)

def test_equal2():
    """
    test broadcast1
    """
    x = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    y = np.array([[1, 2, 3]])
    res = np.equal(x, y)
    obj.run(res=res, x=x, y=y)

def test_equal3():
    """
    test broadcast2
    """
    x = np.array([[[[1, 2, 3], [1, 2, 3], [1, 2, 3]]]])
    y = np.array([[1, 2, 3]])
    res = np.equal(x, y)
    obj.run(res=res, x=x, y=y)

def test_equal4():
    """
    test broadcast3  x.shape < y.shape
    """
    x = np.array([[1, 2, 3]])
    y = np.array([[[[1, 2, 3], [1, 2, 3], [1, 2, 3]]]])
    res = np.equal(x, y)
    obj.run(res=res, x=x, y=y)

def test_equal5():
    """
    x != y
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    y = randtool("float", -10, 10, [3, 3, 3])
    res = np.equal(x, y)
    obj.run(res=res, x=x, y=y)

def test_equal6():
    """
    x != y broadcast
    """
    x = randtool("float", -10, 10, [3, 3, 3, 1])
    y = randtool("float", -10, 10, [3, 3, 1])
    res = np.equal(x, y)
    obj.run(res=res, x=x, y=y)

def test_equal7():
    """
    x != y broadcast  x.shape < y.shape
    """
    x = randtool("float", -10, 10, [3, 3, 1])
    y = randtool("float", -10, 10, [3, 3, 3, 1])
    res = np.equal(x, y)
    obj.run(res=res, x=x, y=y)
