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


class TestLogicalAnd(APIBase):
    """
    test logical_and
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.bool]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False


obj = TestLogicalAnd(paddle.fluid.layers.logical_and)


def test_logical_and_1D_tensor():
    """
    logical_and_1D_tensor
    """
    x_data = np.array([True])
    y_data = np.array([True, False, True, False])
    res = np.logical_and(x_data, y_data)
    obj.run(res=res, x=x_data, y=y_data)


def test_logical_and_broadcast_1():
    """
    logical_and_broadcast_1
    """
    x_data = np.arange(1, 7).reshape((1, 2, 1, 3)).astype(np.bool)
    y_data = np.arange(0, 6).reshape((1, 2, 3)).astype(np.bool)
    res = np.logical_and(x_data, y_data)
    obj.run(res=res, x=x_data, y=y_data)


def test_logical_and_broadcast_2():
    """
    logical_and_broadcast_2
    """
    x_data = np.arange(1, 3).reshape((1, 2)).astype(np.bool)
    y_data = np.arange(0, 4).reshape((2, 2)).astype(np.bool)
    res = np.logical_and(x_data, y_data)
    obj.run(res=res, x=x_data, y=y_data)