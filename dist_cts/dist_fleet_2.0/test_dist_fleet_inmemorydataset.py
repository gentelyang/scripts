#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author:gentelyang  time:2021-06-15
import os
import subprocess
from utils import run_priority

os.system("export CUDA_VISIBLE_DEVICES=0,1")
class TestDistInmemoryDataSetApi():
    """TestDistInmemoryDataSetApi"""
    def test_inmemorydataset1(self):
        """test_inmemorydataset1"""
        cmd = 'fleetrun dist_inmemorydataset1.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0

    def test_inmemorydataset2(self):
        """test_inmemorydataset2"""
        cmd = 'fleetrun dist_inmemorydataset2.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0

    def test_inmemorydataset3(self):
        """test_inmemorydataset3"""
        cmd = 'fleetrun dist_inmemorydataset3.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0

    def test_inmemorydataset4(self):
        """test_inmemorydataset4"""
        cmd = 'fleetrun dist_inmemorydataset4.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0