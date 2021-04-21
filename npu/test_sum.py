#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
*
* Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
* @file test_sum.py
* @author liyang109
* @date 2021/3/23 3:53
* @brief test sum
*
**************************************************************************/
"""
from apibase import APIBase
import paddle
import numpy as np


class TestSum(APIBase):
    """
    test sum
    """
    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        self.enable_backward = False

obj = TestSum(paddle.fluid.layers.sum)


def test_sum_base():
    """
    base,back has diff on npu
    axis=None
    """
    x = np.array([[0.8, 0.4], [0.7, 0.9]])
    res = [[0.8, 0.4],
           [0.7, 0.9]]
    print('res:{}'.format(res))
    obj.run(res=res, x=x)
