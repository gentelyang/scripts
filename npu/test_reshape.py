#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
*
* Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
* @file test_reshape.py
* @author liyang109
* @date 2021/3/23 3:53
* @brief test reshape
*
**************************************************************************/
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestReshape(APIBase):
    """
    test reshape
    """
    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        self.enable_backward = False

obj = TestReshape(paddle.fluid.layers.reshape)

def test_reshape_base():
    """
    base
    """
    x = np.array([[8, 4], [7, 9]])
    shape = [1, 4]
    res = np.reshape(x, shape)
    obj.base(res=res, x=x, shape=shape)

def test_reshape1():
    """
    shape has -1
    """
    x = np.array([[8, 4], [7, 9]])
    shape = [1, -1]
    res = np.reshape(x, shape)
    obj.base(res=res, x=x, shape=shape)