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
  * @file dist_fleet_worker_index.py
  * @author liyang109@baidu.com
  * @date 2021-01-20 13:07
  * @brief 
  *
  **************************************************************************/
"""
import sys
import paddle.distributed.fleet as fleet
from dist_utils import run_priority


fleet.init()

@run_priority(level='P0')
def test_worker_endpoints():
    """test_worker_endpoints"""
    assert fleet.worker_endpoints() == []
    print("{} ... ok".format(sys._getframe().f_code.co_name))



if __name__ == '__main__':
    test_worker_endpoints()

