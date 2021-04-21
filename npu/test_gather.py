#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
*
* Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
* @file test_gather.py
* @author liyang109
* @date 2021/3/23 3:53
* @brief test gather
*
**************************************************************************/
"""
from apibase import APIBase
import paddle
import numpy as np


class TestGather(APIBase):
    """
    test gather
    """
    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64]
        self.enable_backward = False

obj = TestGather(paddle.fluid.layers.gather)


#gather只支持int32和int64
def test_gather1():
    """
    index is int32,xpu not support int64
    """
    x = np.arange(9).reshape(3, 3)
    index = np.array([0, 2, 1, 0]).astype('int32')
    res = np.array([[0, 1, 2], [6, 7, 8], [3, 4, 5], [0, 1, 2]])
    obj.run(res=res, input=x, index=index)