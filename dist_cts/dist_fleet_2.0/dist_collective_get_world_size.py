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
import paddle.distributed as dist
from utils import run_priority


@run_priority(level='P0')
def test_get_world_size():
    """get world size"""
    assert dist.get_world_size() == 2
    print("{} ... ok".format(sys._getframe().f_code.co_name))


if __name__ == '__main__':
    test_get_world_size()