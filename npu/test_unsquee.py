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
  * @file test_unsquee.py
  * @author liyang109@baidu.com
  * @date 2021-03-24 14:52
  * @brief 
  *
  **************************************************************************/
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestUnsqueeze(APIBase):
    """
    test unsqueeze
    """
    def hook(self):
        """
        implement
        """
        self.types = [np.float32]
        self.enable_backward = False
obj = TestUnsqueeze(paddle.fluid.layers.unsqueeze)

def test_unsqueeze_base():
    """
    base
    axis=int32
    Cannot support x as bool tensor, it's a bug
    """
    x = np.arange(6).reshape(2, 3)
    axis = 1
    res = np.expand_dims(x, axis)
    obj.run(res=res, input=x, axes=axis)


def test_unsqueeze2():
    """
    axis=negtive_num
    """
    x = np.arange(6).reshape(2, 3)
    axis = -1
    res = np.expand_dims(x, axis)
    obj.run(res=res, input=x, axes=axis)