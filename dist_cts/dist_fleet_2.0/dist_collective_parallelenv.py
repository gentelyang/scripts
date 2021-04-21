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
import os
import sys
import paddle
import paddle.distributed as dist
from utils import run_priority
os.system("unset CUDA_VISIBLE_DEVICES")
os.system("export CUDA_VISIBLE_DEVICES=1")
dist.init_parallel_env()
parallel_env = dist.ParallelEnv()


@run_priority(level="P0")
def test_get_rank():
    """parallelenv"""
    assert parallel_env.rank == 0
    print("{} ... ok".format(sys._getframe().f_code.co_name))

@run_priority(level="P0")
def test_get_world_size():
    """parallelenv"""
    assert parallel_env.world_size == 1
    print("{} ... ok".format(sys._getframe().f_code.co_name))

@run_priority(level="P0")
def test_get_device_id():
    """parallelenv"""
    assert parallel_env.device_id == 0
    print("{} ... ok".format(sys._getframe().f_code.co_name))

@run_priority(level="P0")
def test_get_current_endpoint():
    """parallelenv"""
    assert str(parallel_env.current_endpoint).startswith("127.0.0.1")
    print("{} ... ok".format(sys._getframe().f_code.co_name))

@run_priority(level="P0")
def test_get_trainer_endpoints():
    """parallelenv"""
    assert len(parallel_env.trainer_endpoints) == 1
    print("{} ... ok".format(sys._getframe().f_code.co_name))


if __name__ == '__main__':
    test_get_rank()
    test_get_world_size()
    test_get_device_id()
    test_get_current_endpoint()
    test_get_trainer_endpoints()
    test_get_trainer_endpoints()
