#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file test_matmul.py
  * @author liyang109
  * @date 2021/3/23 3:53
  * @brief test paddle.matmul
  *
  **************************************************************************/
"""
from apibase import APIBase
import paddle
import numpy as np


class TestMatmul(APIBase):
    """
    test matmul
    """
    def hook(self):
        """
        implement
        """
        self.types = [np.float16]
        self.enable_backward = False

obj = TestMatmul(paddle.fluid.layers.matmul)

def test_matmul_vector_vector():
    """
    vector * vector
    """
    np.random.seed(0)
    x_data = np.random.random([1024,512]).astype(np.float32)
    y_data = np.random.random([512,1024]).astype(np.float32)
    res = np.array([3.58071361])
    obj.run(res=res, x=x_data, y=y_data)