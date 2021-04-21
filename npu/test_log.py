#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
*
* Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
* @file test_log.py
* @author liyang109
* @date 2021/3/23 3:53
* @brief test log
*
**************************************************************************/
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestLog(APIBase):
    """
    test log
    """
    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float16]
        # log has backward compute
        self.enable_backward = False

obj = TestLog(paddle.fluid.layers.log)

#对于x=0的时候应该是负无穷， log7测试case
def test_log3():
    """
    x = large num
    """
    x = np.array([23333, 463333, 665432222])
    res = np.log(x)
    obj.run(res=res, x=x)

def test_log4():
    """
    x = float(tensor)
    """
    x = np.array([0.33332, 0.800002, 0.44444])
    res = np.log(x)
    obj.run(res=res, x=x)

def test_log5():
    """
    x = many dimensions
    """
    x = 1 + np.arange(12).reshape(2, 2, 3)
    res = np.log(x)
    obj.run(res, x=x)

def test_log6():
    """
    name is defined
    """
    x = 1 + np.arange(12).reshape(2, 2, 3)
    res = np.log(x)
    obj.run(res, x=x, name='test_log')

#对于x=0的时候应该是负无穷
def test_log7():
    """
    x=0
    """
    x = np.array([0])
    res = np.array([-np.inf])
    obj.run(res=res, x=x)
