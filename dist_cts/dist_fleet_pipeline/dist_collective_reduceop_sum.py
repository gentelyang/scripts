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
from dist_utils import run_priority
from paddle.distributed import ReduceOp
from paddle.distributed import init_parallel_env

types = [np.float16, np.float32, np.float64, np.int32, np.int64]
paddle.set_device('gpu:%d' % paddle.distributed.ParallelEnv().dev_id)
init_parallel_env()

@run_priority(level='P0')
def test_reduceop_sum():
    """reduceop"""

    for t in types:
        if paddle.distributed.ParallelEnv().local_rank == 0:
            np_data = np.array([[4, 5, 6], [4, 5, 6]]).astype(t)
        else:
            np_data = np.array([[1, 2, 3], [1, 2, 3]]).astype(t)
        data = paddle.to_tensor(np_data)
        paddle.distributed.all_reduce(data, op=ReduceOp.SUM)
        out = data.numpy()
        assert len(out) == 2
        print("test_ReduceOp.SUM %s... ok" % t)


if __name__ == '__main__':
    test_reduceop_sum()