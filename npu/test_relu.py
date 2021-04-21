#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
*
* Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
* @file test_functional_relu.py
* @author liyang109
* @date 2021/3/23 3:53
* @brief test_functional_relu
*
**************************************************************************/
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestFunctionalRelu(APIBase):
    """
    test
    """
    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False
obj = TestFunctionalRelu(paddle.nn.functional.relu)

def test_relu_base():
    """
    base
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    res = np.maximum(0, x)
    obj.run(res=res, x=x)

def test_relu():
    """
    default
    """
    x = randtool("float", -10, 10, [10, 10, 10])
    res = np.maximum(0, x)
    obj.run(res=res, x=x)