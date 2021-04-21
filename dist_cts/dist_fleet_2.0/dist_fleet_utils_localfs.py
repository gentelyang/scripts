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
  * @file dist_fleet_utils_localfs.py
  * @author liyang109@baidu.com
  * @date 2021-01-20 19:22
  * @brief 
  *
  **************************************************************************/
"""
import os
import sys
from paddle.distributed.fleet.utils import LocalFS
from utils import run_priority


client = LocalFS()

@run_priority(level='P1')
def test_is_dir():
    """test_is_dir"""
    subdirs, files = client.ls_dir("./")
    print(subdirs, files)
    print("{} ... ok".format(sys._getframe().f_code.co_name))

@run_priority(level='P1')
def test_mkdirs():
    """test_mkdirs"""
    client.mkdirs("test_mkdirs")
    assert os.path.exists("test_mkdirs") == True
    assert os.path.isdir("test_mkdirs") == True
    print("{} ... ok".format(sys._getframe().f_code.co_name))

@run_priority(level='P1')
def test_delete():
    """test_delete"""
    client.delete("test_mkdirs")
    assert os.path.exists("test_mkdirs") == False
    print("{} ... ok".format(sys._getframe().f_code.co_name))

@run_priority(level='P1')
def test_touch():
    """test_touch"""
    client.touch("test_rename_src")
    assert os.path.isfile("test_rename_src") == True
    print("{} ... ok".format(sys._getframe().f_code.co_name))

@run_priority(level='P1')
def test_rename():
    """test_rename"""
    client.rename("test_rename_src", "test_rename_dst")
    assert os.path.isfile("test_rename_dst")
    print("{} ... ok".format(sys._getframe().f_code.co_name))

@run_priority(level='P1')
def test_is_file():
    """test_is_file"""
    client.touch("is_file")
    client.is_file("is_file")
    print("{} ... ok".format(sys._getframe().f_code.co_name))

@run_priority(level='P1')
def test_is_exist():
    """test_is_exist"""
    client.is_exist("test_rename_dst")
    print("{} ... ok".format(sys._getframe().f_code.co_name))

@run_priority(level='P1')
def test_local_mv():
    """test_local_mv"""
    client.mv("test_rename_dst", "test_mv_dst")
    print("{} ... ok".format(sys._getframe().f_code.co_name))

@run_priority(level='P1')
def test_local_list_dirs():
    """test_local_list_dirs"""
    subdirs = client.list_dirs("./")
    print(subdirs)
    print("{} ... ok".format(sys._getframe().f_code.co_name))


if __name__ == '__main__':
    test_is_dir()
    test_mkdirs()
    test_delete()
    test_touch()
    test_rename()
    test_is_file()
    test_is_exist()
    test_local_mv()
    test_local_list_dirs()