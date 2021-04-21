#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author:gentelyang  time:2021-04-19
import os
import subprocess


class TestTreeNodeAndGrahpEngineApi():
    """test all comm api"""
    def test_tree_node(self):
        """test_tree_node"""
        cmd = 'fleetrun dist_fleet_tree_node.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0

    def test_graph_engine(self):
        """test_tree_node"""
        cmd = 'fleetrun dist_fleet_grahp_engine.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0