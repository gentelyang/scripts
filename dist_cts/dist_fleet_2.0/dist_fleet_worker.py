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
  * @file dist_fleet_worker.py
  * @author liyang109@baidu.com
  * @date 2021-01-20 13:07
  * @brief 
  *
  **************************************************************************/
"""
import sys
import paddle.distributed.fleet as fleet
from utils import run_priority


fleet.init()
@run_priority(level='P0')
def test_is_first_worker():
    """test_is_first_worker"""
    assert fleet.is_first_worker() == True
    print("{} ... ok".format(sys._getframe().f_code.co_name))

@run_priority(level='P0')
def test_worker_index():
    """test_worker_index"""
    assert fleet.worker_index() == 0
    print("{} ... ok".format(sys._getframe().f_code.co_name))

@run_priority(level='P0')
def test_worker_num():
    """test_worker_num"""
    assert fleet.worker_num() == 1
    print("{} ... ok".format(sys._getframe().f_code.co_name))

@run_priority(level='P0')
def test_is_worker():
    """test_is_worker"""
    assert fleet.is_worker() == True
    print("{} ... ok".format(sys._getframe().f_code.co_name))

@run_priority(level='P0')
def test_worker_endpoints():
    """test_worker_endpoints"""
    assert fleet.worker_endpoints() == []
    print("{} ... ok".format(sys._getframe().f_code.co_name))

@run_priority(level='P0')
def test_server_num():
    """test_server_num"""
    assert fleet.server_num() == 0
    print("{} ... ok".format(sys._getframe().f_code.co_name))

@run_priority(level='P0')
def test_server_index():
    """test_server_index"""
    assert fleet.server_index() == 0
    print("{} ... ok".format(sys._getframe().f_code.co_name))

@run_priority(level='P0')
def test_server_endpoints():
    """test_server_endpoints"""
    assert fleet.server_endpoints() == ''
    print("{} ... ok".format(sys._getframe().f_code.co_name))

@run_priority(level='P0')
def test_is_server():
    """test_is_server"""
    assert fleet.is_server() == False
    print("{} ... ok".format(sys._getframe().f_code.co_name))

@run_priority(level='P0')
def test_barrier_worker():
    """test_barrier_worker"""
    assert fleet.barrier_worker() is None
    print("{} ... ok".format(sys._getframe().f_code.co_name))

#fleet_base.py:51: UserWarning: init_worker() function doesn't work when use non_distributed fleet.
@run_priority(level='P0')
def test_init_worker():
    """test_barrier_worker"""
    assert fleet.init_worker() is None
    print("{} ... ok".format(sys._getframe().f_code.co_name))

@run_priority(level='P0')
def test_init_server():
    """test_init_server"""
    assert fleet.init_server() is None
    print("{} ... ok".format(sys._getframe().f_code.co_name))


if __name__ == '__main__':
    test_barrier_worker()
    test_init_worker()
    test_is_first_worker()
    test_is_server()
    test_is_worker()
    test_server_endpoints()
    test_server_index()
    test_server_num()
    test_worker_endpoints()
    test_worker_num()
    test_worker_index()

