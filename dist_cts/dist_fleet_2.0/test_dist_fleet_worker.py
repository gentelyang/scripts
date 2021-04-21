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
  * @file test_dist_fleet_worker.py
  * @author liyang109@baidu.com
  * @date 2021-01-20 15:01
  * @brief 
  *
  **************************************************************************/
"""
import os
import subprocess
from utils import run_priority


class TestFleetWorkerServerApi():
    """TestFleetWorkerServerApi"""
    def test_dist_fleet_worker(self):
        """test_dist_fleet_worker"""
        cmd = 'fleetrun dist_fleet_worker.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0

