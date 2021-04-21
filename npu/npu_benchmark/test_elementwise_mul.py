#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
*
* Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
* @file test_elementwise_mul.py
* @author liyang109
* @date 2021/3/23 3:53
* @brief test_elementwise_mul
*
**************************************************************************/
"""
from apibase import APIBase
from apibase import randtool
import paddle
import numpy as np


class TestElementwiseMul(APIBase):
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

obj = TestElementwiseMul(paddle.fluid.layers.elementwise_mul)

def test_elementwise_mul():
    """
    default
    """
    x = randtool("float", -10, 10, [1])
    y = randtool("float", -10, 10, [1232, 19000])
#    res = np.multiply(x, y)
    res = [1]
    print(res)
    obj.run(res=res, x=x, y=y)