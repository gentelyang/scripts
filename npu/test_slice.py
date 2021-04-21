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
  * @file test_slice.py
  * @author liyang109@baidu.com
  * @date 2021-03-24 14:59
  * @brief
  *
  **************************************************************************/
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
File: test_slice.py
Authors: liyang109
Date: 2021/3/23 3:53
"""
from apibase import APIBase
from apibase import randtool
import paddle
import numpy as np


class TestSlice(APIBase):
    """
    test slice
    """
    def hook(self):
        """
        implement
        """
        self.types = [np.float32]
        # self.debug = True
        # enable check grad
        self.enable_backward = False

obj = TestSlice(paddle.fluid.layers.slice)

def test_slice_base():
    """
    base,x=float,axes,starts,ends=list
    """
    x = np.array([[1, 2, 3, 4], [5, 6, 7, 8], ])
    axes = [0, 1]
    starts = [1, 0]
    ends = [2, 3]
    res = np.array([[5, 6, 7], ])
    obj.run(res=res, input=x, axes=axes, starts=starts, ends=ends)

def test_slice1():
    """
    base,axes,starts,ends=tuple
    """
    x = np.array([[1, 2, 3, 4], [5, 6, 7, 8], ])
    axes = (0, 1)
    starts = (1, 0)
    ends = (2, 3)
    res = np.array([[5, 6, 7], ])
    obj.run(res=res, input=x, axes=axes, starts=starts, ends=ends)

#starts、end是np.array时会报错
# def test_slice2():
#     """
#     base,axes=tuple,starts,ends=Tensor
#     """
#     t = np.float32
#     x = np.array([[1, 2, 3, 4], [5, 6, 7, 8], ]).astype(t)
#     axes = (0, 1)
#     starts = np.array([1, 0])
#     ends = np.array([2, 3])
#     res = np.array([[5, 6, 7], ]).astype(t)
#     obj.run(res=res, input=x, axes=axes, starts=starts, ends=ends)

def test_slice3():
    """
    base,axes,starts,ends=tuple
    """
    x = np.array([[1, 2, 3, 4], [5, 6, 7, 8], ])
    axes = (0, 1)
    starts = (0, 1)
    ends = (-1, 1000)
    res = np.array([[2, 3, 4], ])
    obj.run(res=res, input=x, axes=axes, starts=starts, ends=ends)