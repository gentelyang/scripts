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
import numpy as np
import paddle
from utils import run_priority
from paddle.distributed import init_parallel_env

types = [np.float16, np.float32, np.float64, np.int32, np.int64]

@run_priority(level='P0')
def test_scatter():
    """scatter"""
    paddle.set_device('gpu:%d' % paddle.distributed.ParallelEnv().dev_id)
    init_parallel_env()
    for t in types:
        if paddle.distributed.ParallelEnv().local_rank == 0:
            np_data1 = np.array([7, 8, 9]).astype(t)
            np_data2 = np.array([10, 11, 12]).astype(t)
        else:
            np_data1 = np.array([1, 2, 3]).astype(t)
            np_data2 = np.array([4, 5, 6]).astype(t)
        data1 = paddle.to_tensor(np_data1)
        data2 = paddle.to_tensor(np_data2)
        if paddle.distributed.ParallelEnv().local_rank == 0:
            paddle.distributed.scatter(data1, src=1)
        else:
            paddle.distributed.scatter(data1, tensor_list=[data1, data2], src=1)
        out = data1.numpy()
        assert len(out) == 3
        print("test_scatter %s ... ok" % t)


if __name__ == '__main__':
    test_scatter()