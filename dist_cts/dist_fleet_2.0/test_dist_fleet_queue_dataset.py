#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author:gentelyang  time:2021-06-15
import os
import subprocess
from utils import run_priority

os.system("export CUDA_VISIBLE_DEVICES=0,1")
class TestDistQueueDataSetApi():
    """TestDistQueueDataSetApi"""
    def test_queuedataset1(self):
        """test_queuedataset1"""
        cmd = 'fleetrun dist_queuedataset1.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0

    def test_queuedataset2(self):
        """test_queuedataset2"""
        cmd = 'fleetrun dist_queuedataset2.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0