#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
*
* Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
* @file test_pow.py
* @author liyang109
* @date 2021/3/23 3:53
* @brief
*
**************************************************************************/
"""
from apibase import APIBase
from apibase import randtool
import paddle
import numpy as np


class TestPow(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64, np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False


obj = TestPow(paddle.fluid.layers.pow)


def test_pow1():
    """
    y = 0
    """
    x = randtool("float", 1, 2, [2, 2, 2])
    y = 0
    res = np.power(x, y)
    obj.run(res=res, x=x, factor=y)


