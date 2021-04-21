#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
*
* Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
* @file test_elementwise_sub.py
* @author liyang109
* @date 2021/3/23 3:53
* @brief test_elementwise_add
*
**************************************************************************/
"""
from apibase import APIBase
from apibase import randtool
import paddle
import numpy as np


class TestElementwiseAdd(APIBase):
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

obj = TestElementwiseAdd(paddle.fluid.layers.elementwise_add)

def test_elementwise_add():
    """
    default
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    y = randtool("float", -10, 10, [3, 3, 3])
    res = x + y
    print(res)
    obj.run(res=res, x=x, y=y)