#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file:  test_transpose.py
  * @author:  liyang109
  * @date  2021/3/23 3:53 PM
  * @brief
  *
  **************************************************************************/
"""
from apibase import APIBase
from apibase import randtool
import paddle
import numpy as np
import math

class TestTranspose(APIBase):
    """
    test transpose
    """
    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64, np.int32, np.int64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False

obj = TestTranspose(paddle.fluid.layers.transpose)

def test_transpose_base():
    """
    x.shape=(1, 2, 2, 3)
    res.shape=(1, 3, 2, 2)
    float32
    """
    x = np.ones([1, 2, 2, 3]).astype(np.float32)
    perm = [0, 3, 1, 2]
    res = np.array([[[[1, 1],
                      [1, 1]],
                     [[1, 1],
                      [1, 1]],
                     [[1, 1],
                      [1, 1]]]]).astype(np.float32)
    obj.run(res=res, x=x, perm=perm)

def test_transpose1():
    """
    x.shape=(1, 2, 2, 3)
    res.shape=(1, 3, 2, 2)
    int32
    """
    x = np.ones([1, 2, 2, 3]).astype(np.int32)
    perm = [0, 3, 1, 2]
    res = np.array([[[[1, 1],
                      [1, 1]],
                     [[1, 1],
                      [1, 1]],
                     [[1, 1],
                      [1, 1]]]]).astype(np.int32)
    obj.run(res=res, x=x, perm=perm)