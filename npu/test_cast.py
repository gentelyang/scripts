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
* @brief test_cast
*
**************************************************************************/
"""
from apibase import APIBase
from apibase import randtool
import paddle
import numpy as np


class TestElementwiseSub(APIBase):
    """
    test
    """
    def hook(self):
        """
        implement
        """
        self.types = [np.float16]
        self.debug = False
        # self.static = True
        # enable check grad
        self.enable_backward = False

obj = TestElementwiseSub(paddle.fluid.layers.cast)

def test_cast():
    """
    default
    """
    x = np.ones([2, 3])
    dtype = np.float32
    res = x.astype(dtype)
    obj.run(res=res, x=x, dtype=dtype)