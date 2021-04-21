#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expnottab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file test_logical_not.py
  * @author liyang109
  * @date 2020/09/25 16:00
  * @brief test paddle.logical_not
  *
  **************************************************************************/
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestLogicalNot(APIBase):
    """
    test logical_not
    """
    def hook(self):
        """
        implement
        """
        self.types = [np.bool]
        #self.debug = True
        #self.static = True
        # enable check grad
        self.enable_backward = False

obj = TestLogicalNot(paddle.fluid.layers.logical_not)

def test_logical_not_1D_tensor():
    """
    logical_not_1D_tensor
    """
    x_data = np.array([True])
    res = np.logical_not(x_data)
    obj.run(res=res, x=x_data)

def test_logical_not_broadcast_1():
    """
    logical_not_broadcast_1
    """
    x_data = np.arange(1, 7).reshape((1, 2, 1, 3)).astype(np.bool)
    res = np.logical_not(x_data)
    obj.run(res=res, x=x_data)

def test_logical_not_broadcast_2():
    """
    logical_not_broadcast_2
    """
    x_data = np.arange(1, 3).reshape((1, 2)).astype(np.bool)
    res = np.logical_not(x_data)
    obj.run(res=res, x=x_data)