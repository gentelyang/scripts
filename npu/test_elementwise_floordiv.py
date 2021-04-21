#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
#======================================================================
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
#======================================================================
"""
/***************************************************************************
  *
  * Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
  * @file test_elementwise_floordiv.py
  * @author liyang109@baidu.com
  * @date 2021-03-24 16:18
  * @brief 
  *
  **************************************************************************/
"""
from apibase import APIBase
from apibase import randtool
import paddle
import numpy as np


class TestElementwiseDiv(APIBase):
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

obj = TestElementwiseDiv(paddle.fluid.layers.elementwise_floordiv)

#目前cpu上也报错
def test_elementwise_floordiv():
    """
    default
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    y = randtool("float", -10, 10, [3, 3, 3])
    res = np.divide(x, y)
    print(res)
    obj.run(res=res, x=x, y=y)