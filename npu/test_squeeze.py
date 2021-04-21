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
  * @file test_squeeze.py
  * @author liyang109@baidu.com
  * @date 2021-03-24 15:19
  * @brief 
  *
  **************************************************************************/
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestSqueeze(APIBase):
    """
    test squeeze
    """
    def hook(self):
        """
        implement
        """
        self.types = [np.float32]
        #self.debug = True
        #self.static = True
        # enable check grad
        self.enable_backward = False

obj = TestSqueeze(paddle.fluid.layers.squeeze)

#res = np.array([[0., 1., 2.], [3., 4., 5.]])应该是这个维度，目前是npu输出是多一维的
def test_squeeze_base():
    """
    squeeze_base
    """
    x_data = np.arange(6).reshape((1, 2, 1, 3)).astype(np.float32)
    res = np.array([[[0., 1., 2.]], [[3., 4., 5.]]])
    obj.run(res=res, input=x_data, axes=[-4])

def test_squeeze_axis1():
    """
    axis = None
    """
    x_data = np.arange(6).reshape((1, 2, 1, 3)).astype(np.float32)
    res = np.array([[[0., 1., 2.]], [[3., 4., 5.]]])
    obj.run(res=res, input=x_data, axes=[-4])
