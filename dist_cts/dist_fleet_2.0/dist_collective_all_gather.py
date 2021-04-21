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
  * @file test.py
  * @author liyang109@baidu.com
  * @date 2020-12-30 15:53
  * @brief
  *
  **************************************************************************/
"""
import sys
import numpy as np
import paddle
from paddle.distributed import init_parallel_env
from utils import run_priority


paddle.set_device('gpu:%d' % paddle.distributed.ParallelEnv().dev_id)
init_parallel_env()

@run_priority(level='P0')
def test_all_gather():
    """all gather"""
    tensor_list = []
    types=[np.float16, np.float32, np.float64, np.int32, np.int64]
    for t in types:
       if paddle.distributed.ParallelEnv().local_rank == 0:
           np_data1 = np.array([[4, 5, 6], [4, 5, 6]]).astype(t)
           np_data2 = np.array([[4, 5, 6], [4, 5, 6]]).astype(t)
           data1 = paddle.to_tensor(np_data1)
           data2 = paddle.to_tensor(np_data2)
           paddle.distributed.all_gather(tensor_list, data1)
       else:
           np_data1 = np.array([[1, 2, 3], [1, 2, 3]]).astype(t)
           np_data2 = np.array([[1, 2, 3], [1, 2, 3]]).astype(t)
           data1 = paddle.to_tensor(np_data1)
           data2 = paddle.to_tensor(np_data2)
           paddle.distributed.all_gather(tensor_list, data2)
       assert len(tensor_list) == 2
       print("{} ... ok".format(sys._getframe().f_code.co_name))


if __name__ == '__main__':
    test_all_gather()