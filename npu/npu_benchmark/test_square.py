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
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file test_square.py
  * @author liyang109@baidu.com
  * @date 2021/3/23 3:53
  * @brief
  *
  **************************************************************************/
"""
from apibase import APIBase
import paddle
import numpy as np


class TestSquare(APIBase):
    """
    test paddle.square api
    """
    def hook(self):
        """
        implement
        """
        self.types = [np.float16]
        self.enable_backward = False
obj = TestSquare(paddle.fluid.layers.square)

def test_base():
    """
    base
    :return: Tensor
    """
    x = np.random.random([3072, 768]).astype(np.float16)
    res = np.square(x)
    obj.run(res=res, x=x)