#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
*
* Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
* @file test_elementwise_pow.py
* @author liyang109
* @date 2021/3/23 3:53
* @brief test_elementwise_pow
*
**************************************************************************/
"""
from apibase import APIBase
from apibase import randtool
import paddle
import numpy as np


class TestElementwisePow(APIBase):
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

obj = TestElementwisePow(paddle.fluid.layers.elementwise_pow)

def test_elementwise_pow():
    """
    default
    """
    x = randtool("float", 1, 2, [2, 2, 2])
    y = randtool("int", 1, 2, [2, 2, 2])
    res = np.power(x, y)
    obj.run(res=res, x=x, y=y)

def test_pow1():
    """
    y = 0
    """
    x = randtool("float", 1, 2, [2, 2, 2])
    y = randtool("int", 2, 3, [1, 1, 1])
    res = np.power(x, y)
    obj.run(res=res, x=x, y=y)