#!/usr/bin/env python
#-*- coding:utf-8 -*-
# @Time    : 2019-09-12 15:07
# @Author  : liyang109

from __future__ import print_function
import nose.tools as tools
from dist_base_fleet import TestFleetBase
import json
from dist_base_fleet import run_by_freq


class TestDistSimnet(TestFleetBase):
    def __init__(self):
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.single_cpu_data = [0.09485437, 0.08698994, 0.08731181, 0.10150592, 0.10547037]
        self._model_file = 'dist_fleet_simnet_bow.py'

    def check_data(self, loss, delta=None, expect=None):
        """
        校验结果数据
        """
        if expect:
            expect_data = expect
        else:
            expect_data = self.single_cpu_data
        if delta:
            for i in range(len(expect_data)):
                tools.assert_almost_equal(loss[i], expect_data[i], delta=delta)
        else:
            for i in range(len(expect_data)):
                tools.assert_equal(loss[i], expect_data[i])
    """async"""
    def test_sim_2ps_2tr_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode':'async','cpu_num':2,'num_threads':2,
                           'slice_var_up':True,'enable_dc_asgd':False, 'split_method':False,
                           'runtime_split_send_recv':True,'geo_sgd':True,'wait_port':True,
                           'use_hierarchical_allreduce':False,'push_nums':25}
        self.test_sim_2ps_2tr_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_sim_2ps_2tr_async_2thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_async_2thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode':'async','cpu_num':2,'num_threads':2,
                           'slice_var_up':False,'enable_dc_asgd':False, 'split_method':False,
                           'runtime_split_send_recv':True,'geo_sgd':True,'wait_port':True,
                           'use_hierarchical_allreduce':False,'push_nums':25}
        self.test_sim_2ps_2tr_async_2thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_async_2thread_Tslice_Tdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_async_2thread_Tslice_Tdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode':'async','cpu_num':2,'num_threads':2,
                           'slice_var_up':True,'enable_dc_asgd':False, 'split_method':False,
                           'runtime_split_send_recv':True,'geo_sgd':True,'wait_port':True,
                           'use_hierarchical_allreduce':False,'push_nums':25}
        self.test_sim_2ps_2tr_async_2thread_Tslice_Tdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_sim_2ps_2tr_async_2thread_Fslice_Tdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_async_2thread_Fslice_Tdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode':'async','cpu_num':2,'num_threads':2,
                           'slice_var_up':False,'enable_dc_asgd':False, 'split_method':False,
                           'runtime_split_send_recv':True,'geo_sgd':True,'wait_port':True,
                           'use_hierarchical_allreduce':False,'push_nums':25}
        self.test_sim_2ps_2tr_async_2thread_Fslice_Tdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_async_2thread_Tslice_Fdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_async_2thread_Tslice_Fdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode':'async','cpu_num':2,'num_threads':2,
                           'slice_var_up':True,'enable_dc_asgd':False, 'split_method':True,
                           'runtime_split_send_recv':True,'geo_sgd':True,'wait_port':True,
                           'use_hierarchical_allreduce':False,'push_nums':25}
        self.test_sim_2ps_2tr_async_2thread_Tslice_Fdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_async_2thread_Fslice_Fdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_async_2thread_Fslice_Fdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode':'async','cpu_num':2,'num_threads':2,
                           'slice_var_up':False,'enable_dc_asgd':False, 'split_method':True,
                           'runtime_split_send_recv':True,'geo_sgd':True,'wait_port':True,
                           'use_hierarchical_allreduce':False,'push_nums':25}
        self.test_sim_2ps_2tr_async_2thread_Fslice_Fdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_sim_2ps_2tr_async_2thread_Tslice_Tdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_async_2thread_Tslice_Tdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode':'async','cpu_num':2,'num_threads':2,
                           'slice_var_up':True,'enable_dc_asgd':False, 'split_method':True,
                           'runtime_split_send_recv':True,'geo_sgd':True,'wait_port':True,
                           'use_hierarchical_allreduce':False,'push_nums':25}
        self.test_sim_2ps_2tr_async_2thread_Tslice_Tdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_sim_2ps_2tr_async_2thread_Fslice_Tdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_async_2thread_Fslice_Tdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode':'async','cpu_num':2,'num_threads':2,
                           'slice_var_up':False,'enable_dc_asgd':False, 'split_method':True,
                           'runtime_split_send_recv':True,'geo_sgd':True,'wait_port':True,
                           'use_hierarchical_allreduce':False,'push_nums':25}
        self.test_sim_2ps_2tr_async_2thread_Fslice_Tdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_sim_2ps_2tr_async_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_async_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode':'async','cpu_num':2,'num_threads':2,
                           'slice_var_up':True,'enable_dc_asgd':False, 'split_method':False,
                           'runtime_split_send_recv':True,'geo_sgd':False,'wait_port':True,
                           'use_hierarchical_allreduce':False,'push_nums':25}
        self.test_sim_2ps_2tr_async_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_async_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_async_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode':'async','cpu_num':2,'num_threads':2,
                           'slice_var_up':False,'enable_dc_asgd':False, 'split_method':False,
                           'runtime_split_send_recv':True,'geo_sgd':False,'wait_port':True,
                           'use_hierarchical_allreduce':False,'push_nums':25}
        self.test_sim_2ps_2tr_async_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_async_2thread_Tslice_Tdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_async_2thread_Tslice_Tdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode':'async','cpu_num':2,'num_threads':2,
                           'slice_var_up':True,'enable_dc_asgd':False, 'split_method':False,
                           'runtime_split_send_recv':True,'geo_sgd':False,'wait_port':True,
                           'use_hierarchical_allreduce':False,'push_nums':25}
        self.test_sim_2ps_2tr_async_2thread_Tslice_Tdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_sim_2ps_2tr_async_2thread_Fslice_Tdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_async_2thread_Fslice_Tdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode':'async','cpu_num':2,'num_threads':2,
                           'slice_var_up':False,'enable_dc_asgd':False, 'split_method':False,
                           'runtime_split_send_recv':True,'geo_sgd':False,'wait_port':True,
                           'use_hierarchical_allreduce':False,'push_nums':25}
        self.test_sim_2ps_2tr_async_2thread_Fslice_Tdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_async_2thread_Tslice_Fdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_async_2thread_Tslice_Fdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode':'async','cpu_num':2,'num_threads':2,
                           'slice_var_up':True,'enable_dc_asgd':False, 'split_method':True,
                           'runtime_split_send_recv':True,'geo_sgd':False,'wait_port':True,
                           'use_hierarchical_allreduce':False,'push_nums':25}
        self.test_sim_2ps_2tr_async_2thread_Tslice_Fdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_sim_2ps_2tr_async_2thread_Fslice_Fdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_async_2thread_Fslice_Fdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode':'async','cpu_num':2,'num_threads':2,
                           'slice_var_up':False,'enable_dc_asgd':False, 'split_method':True,
                           'runtime_split_send_recv':True,'geo_sgd':False,'wait_port':True,
                           'use_hierarchical_allreduce':False,'push_nums':25}
        self.test_sim_2ps_2tr_async_2thread_Fslice_Fdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_async_2thread_Tslice_Tdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_async_2thread_Tslice_Tdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode':'async','cpu_num':2,'num_threads':2,
                           'slice_var_up':True,'enable_dc_asgd':False, 'split_method':True,
                           'runtime_split_send_recv':True,'geo_sgd':False,'wait_port':True,
                           'use_hierarchical_allreduce':False,'push_nums':25}
        self.test_sim_2ps_2tr_async_2thread_Tslice_Tdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_async_2thread_Fslice_Tdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_async_2thread_Fslice_Tdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode':'async','cpu_num':2,'num_threads':2,
                           'slice_var_up':False,'enable_dc_asgd':False, 'split_method':True,
                           'runtime_split_send_recv':True,'geo_sgd':False,'wait_port':True,
                           'use_hierarchical_allreduce':False,'push_nums':25}
        self.test_sim_2ps_2tr_async_2thread_Fslice_Tdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode':'async','cpu_num':2,'num_threads':2,
                           'slice_var_up':True,'enable_dc_asgd':False, 'split_method':False,
                           'runtime_split_send_recv':True,'geo_sgd':True,'wait_port':True,
                           'use_hierarchical_allreduce':True,'push_nums':25}
        self.test_sim_2ps_2tr_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_async_2thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_async_2thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode':'async','cpu_num':2,'num_threads':2,
                           'slice_var_up':False,'enable_dc_asgd':False, 'split_method':False,
                           'runtime_split_send_recv':True,'geo_sgd':True,'wait_port':True,
                           'use_hierarchical_allreduce':True,'push_nums':25}
        self.test_sim_2ps_2tr_async_2thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_sim_2ps_2tr_async_2thread_Tslice_Tdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_async_2thread_Tslice_Tdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode':'async','cpu_num':2,'num_threads':2,
                           'slice_var_up':True,'enable_dc_asgd':False, 'split_method':False,
                           'runtime_split_send_recv':True,'geo_sgd':True,'wait_port':True,
                           'use_hierarchical_allreduce':True,'push_nums':25}
        self.test_sim_2ps_2tr_async_2thread_Tslice_Tdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_async_2thread_Fslice_Tdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_async_2thread_Fslice_Tdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode':'async','cpu_num':2,'num_threads':2,
                           'slice_var_up':False,'enable_dc_asgd':False, 'split_method':False,
                           'runtime_split_send_recv':True,'geo_sgd':True,'wait_port':True,
                           'use_hierarchical_allreduce':True,'push_nums':25}
        self.test_sim_2ps_2tr_async_2thread_Fslice_Tdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_sim_2ps_2tr_async_2thread_Tslice_Fdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_async_2thread_Tslice_Fdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode':'async','cpu_num':2,'num_threads':2,
                           'slice_var_up':True,'enable_dc_asgd':False, 'split_method':True,
                           'runtime_split_send_recv':True,'geo_sgd':True,'wait_port':True,
                           'use_hierarchical_allreduce':True,'push_nums':25}
        self.test_sim_2ps_2tr_async_2thread_Tslice_Fdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_async_2thread_Fslice_Fdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_async_2thread_Fslice_Fdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode':'async','cpu_num':2,'num_threads':2,
                           'slice_var_up':False,'enable_dc_asgd':False, 'split_method':True,
                           'runtime_split_send_recv':True,'geo_sgd':True,'wait_port':True,
                           'use_hierarchical_allreduce':True,'push_nums':25}
        self.test_sim_2ps_2tr_async_2thread_Fslice_Fdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_async_2thread_Tslice_Tdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_async_2thread_Tslice_Tdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode':'async','cpu_num':2,'num_threads':2,
                           'slice_var_up':True,'enable_dc_asgd':False, 'split_method':True,
                           'runtime_split_send_recv':True,'geo_sgd':True,'wait_port':True,
                           'use_hierarchical_allreduce':True,'push_nums':25}
        self.test_sim_2ps_2tr_async_2thread_Tslice_Tdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_async_2thread_Fslice_Tdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_async_2thread_Fslice_Tdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode':'async','cpu_num':2,'num_threads':2,
                           'slice_var_up':False,'enable_dc_asgd':False, 'split_method':True,
                           'runtime_split_send_recv':True,'geo_sgd':True,'wait_port':True,
                           'use_hierarchical_allreduce':True,'push_nums':25}
        self.test_sim_2ps_2tr_async_2thread_Fslice_Tdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_sim_2ps_2tr_async_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_async_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode':'async','cpu_num':2,'num_threads':2,
                           'slice_var_up':True,'enable_dc_asgd':False, 'split_method':False,
                           'runtime_split_send_recv':True,'geo_sgd':False,'wait_port':True,
                           'use_hierarchical_allreduce':True,'push_nums':25}
        self.test_sim_2ps_2tr_async_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_sim_2ps_2tr_async_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_async_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode':'async','cpu_num':2,'num_threads':2,
                           'slice_var_up':False,'enable_dc_asgd':False, 'split_method':False,
                           'runtime_split_send_recv':True,'geo_sgd':False,'wait_port':True,
                           'use_hierarchical_allreduce':True,'push_nums':25}
        self.test_sim_2ps_2tr_async_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_async_2thread_Tslice_Tdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_async_2thread_Tslice_Tdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode':'async','cpu_num':2,'num_threads':2,
                           'slice_var_up':True,'enable_dc_asgd':False, 'split_method':False,
                           'runtime_split_send_recv':True,'geo_sgd':False,'wait_port':True,
                           'use_hierarchical_allreduce':True,'push_nums':25}
        self.test_sim_2ps_2tr_async_2thread_Tslice_Tdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_sim_2ps_2tr_async_2thread_Fslice_Tdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_async_2thread_Fslice_Tdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode':'async','cpu_num':2,'num_threads':2,
                           'slice_var_up':False,'enable_dc_asgd':False, 'split_method':False,
                           'runtime_split_send_recv':True,'geo_sgd':False,'wait_port':True,
                           'use_hierarchical_allreduce':True,'push_nums':25}
        self.test_sim_2ps_2tr_async_2thread_Fslice_Tdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_async_2thread_Tslice_Fdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_async_2thread_Tslice_Fdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode':'async','cpu_num':2,'num_threads':2,
                           'slice_var_up':True,'enable_dc_asgd':False, 'split_method':True,
                           'runtime_split_send_recv':True,'geo_sgd':False,'wait_port':True,
                           'use_hierarchical_allreduce':True,'push_nums':25}
        self.test_sim_2ps_2tr_async_2thread_Tslice_Fdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_sim_2ps_2tr_async_2thread_Fslice_Fdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_async_2thread_Fslice_Fdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode':'async','cpu_num':2,'num_threads':2,
                           'slice_var_up':False,'enable_dc_asgd':False, 'split_method':True,
                           'runtime_split_send_recv':True,'geo_sgd':False,'wait_port':True,
                           'use_hierarchical_allreduce':True,'push_nums':25}
        self.test_sim_2ps_2tr_async_2thread_Fslice_Fdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_async_2thread_Tslice_Tdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_async_2thread_Tslice_Tdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode':'async','cpu_num':2,'num_threads':2,
                           'slice_var_up':True,'enable_dc_asgd':False, 'split_method':True,
                           'runtime_split_send_recv':True,'geo_sgd':False,'wait_port':True,
                           'use_hierarchical_allreduce':True,'push_nums':25}
        self.test_sim_2ps_2tr_async_2thread_Tslice_Tdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_sim_2ps_2tr_async_2thread_Fslice_Tdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_async_2thread_Fslice_Tdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode':'async','cpu_num':2,'num_threads':2,
                           'slice_var_up':False,'enable_dc_asgd':False, 'split_method':True,
                           'runtime_split_send_recv':True,'geo_sgd':False,'wait_port':True,
                           'use_hierarchical_allreduce':True,'push_nums':25}
        self.test_sim_2ps_2tr_async_2thread_Fslice_Tdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    """geo_async"""
    @run_by_freq(freq="MONTH")
    def test_sim_2ps_2tr_geo_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_geo_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'geo_async', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': True, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 25}
        self.test_sim_2ps_2tr_geo_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_geo_async_2thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_geo_async_2thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'geo_async', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': True, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 25}
        self.test_sim_2ps_2tr_geo_async_2thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_sim_2ps_2tr_geo_async_2thread_Tslice_Tdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_geo_async_2thread_Tslice_Tdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'geo_async', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': True, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 25}
        self.test_sim_2ps_2tr_geo_async_2thread_Tslice_Tdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_geo_async_2thread_Fslice_Tdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_geo_async_2thread_Fslice_Tdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'geo_async', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': True, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 25}
        self.test_sim_2ps_2tr_geo_async_2thread_Fslice_Tdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_sim_2ps_2tr_geo_async_2thread_Tslice_Fdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_geo_async_2thread_Tslice_Fdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'geo_async', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': True,
                           'runtime_split_send_recv': True, 'geo_sgd': True, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 25}
        self.test_sim_2ps_2tr_geo_async_2thread_Tslice_Fdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_geo_async_2thread_Fslice_Fdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_geo_async_2thread_Fslice_Fdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'geo_async', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': True,
                           'runtime_split_send_recv': True, 'geo_sgd': True, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 25}
        self.test_sim_2ps_2tr_geo_async_2thread_Fslice_Fdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_sim_2ps_2tr_geo_async_2thread_Tslice_Tdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_geo_async_2thread_Tslice_Tdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'geo_async', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': True,
                           'runtime_split_send_recv': True, 'geo_sgd': True, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 25}
        self.test_sim_2ps_2tr_geo_async_2thread_Tslice_Tdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_geo_async_2thread_Fslice_Tdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_geo_async_2thread_Fslice_Tdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'geo_async', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': True,
                           'runtime_split_send_recv': True, 'geo_sgd': True, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 25}
        self.test_sim_2ps_2tr_geo_async_2thread_Fslice_Tdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_geo_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_geo_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'geo_async', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': True, 'wait_port': True,
                           'use_hierarchical_allreduce': True, 'push_nums': 25}
        self.test_sim_2ps_2tr_geo_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_sim_2ps_2tr_geo_async_2thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_geo_async_2thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'geo_async', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': True, 'wait_port': True,
                           'use_hierarchical_allreduce': True, 'push_nums': 25}
        self.test_sim_2ps_2tr_geo_async_2thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_geo_async_2thread_Tslice_Tdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_geo_async_2thread_Tslice_Tdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'geo_async', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': True, 'wait_port': True,
                           'use_hierarchical_allreduce': True, 'push_nums': 25}
        self.test_sim_2ps_2tr_geo_async_2thread_Tslice_Tdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_sim_2ps_2tr_geo_async_2thread_Fslice_Tdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_geo_async_2thread_Fslice_Tdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'geo_async', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': True, 'wait_port': True,
                           'use_hierarchical_allreduce': True, 'push_nums': 25}
        self.test_sim_2ps_2tr_geo_async_2thread_Fslice_Tdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_geo_async_2thread_Tslice_Fdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_geo_async_2thread_Tslice_Fdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'geo_async', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': True,
                           'runtime_split_send_recv': True, 'geo_sgd': True, 'wait_port': True,
                           'use_hierarchical_allreduce': True, 'push_nums': 25}
        self.test_sim_2ps_2tr_geo_async_2thread_Tslice_Fdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_geo_async_2thread_Fslice_Fdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_geo_async_2thread_Fslice_Fdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'geo_async', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': True,
                           'runtime_split_send_recv': True, 'geo_sgd': True, 'wait_port': True,
                           'use_hierarchical_allreduce': True, 'push_nums': 25}
        self.test_sim_2ps_2tr_geo_async_2thread_Fslice_Fdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_sim_2ps_2tr_geo_async_2thread_Tslice_Tdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_geo_async_2thread_Tslice_Tdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'geo_async', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': True,
                           'runtime_split_send_recv': True, 'geo_sgd': True, 'wait_port': True,
                           'use_hierarchical_allreduce': True, 'push_nums': 25}
        self.test_sim_2ps_2tr_geo_async_2thread_Tslice_Tdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_geo_async_2thread_Fslice_Tdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_geo_async_2thread_Fslice_Tdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'geo_async', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': True,
                           'runtime_split_send_recv': True, 'geo_sgd': True, 'wait_port': True,
                           'use_hierarchical_allreduce': True, 'push_nums': 25}
        self.test_sim_2ps_2tr_geo_async_2thread_Fslice_Tdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    """half_async"""
    def test_sim_2ps_2tr_hasync_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_hasync_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'half_async', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 25}
        self.test_sim_2ps_2tr_hasync_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_hasync_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_hasync_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'half_async', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 25}
        self.test_sim_2ps_2tr_hasync_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_sim_2ps_2tr_hasync_2thread_Tslice_Tdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_hasync_2thread_Tslice_Tdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'half_async', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 25}
        self.test_sim_2ps_2tr_hasync_2thread_Tslice_Tdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_hasync_2thread_Fslice_Tdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_hasync_2thread_Fslice_Tdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'half_async', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 25}
        self.test_sim_2ps_2tr_hasync_2thread_Fslice_Tdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_sim_2ps_2tr_hasync_2thread_Tslice_Fdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_hasync_2thread_Tslice_Fdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'half_async', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': True,
                           'runtime_split_send_recv': True, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 25}
        self.test_sim_2ps_2tr_hasync_2thread_Tslice_Fdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_hasync_2thread_Fslice_Fdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_hasync_2thread_Fslice_Fdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'half_async', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': True,
                           'runtime_split_send_recv': True, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 25}
        self.test_sim_2ps_2tr_hasync_2thread_Fslice_Fdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_sim_2ps_2tr_hasync_2thread_Tslice_Tdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_hasync_2thread_Tslice_Tdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'half_async', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': True,
                           'runtime_split_send_recv': True, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 25}
        self.test_sim_2ps_2tr_hasync_2thread_Tslice_Tdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_hasync_2thread_Fslice_Tdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_hasync_2thread_Fslice_Tdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'half_async', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': True,
                           'runtime_split_send_recv': True, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 25}
        self.test_sim_2ps_2tr_hasync_2thread_Fslice_Tdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_sim_2ps_2tr_hasync_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_hasync_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'half_async', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': True, 'push_nums': 25}
        self.test_sim_2ps_2tr_hasync_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_hasync_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_hasync_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'half_async', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': True, 'push_nums': 25}
        self.test_sim_2ps_2tr_hasync_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_hasync_2thread_Tslice_Tdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_hasync_2thread_Tslice_Tdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'half_async', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': True, 'push_nums': 25}
        self.test_sim_2ps_2tr_hasync_2thread_Tslice_Tdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_sim_2ps_2tr_hasync_2thread_Fslice_Tdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_hasync_2thread_Fslice_Tdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'half_async', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': True, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': True, 'push_nums': 25}
        self.test_sim_2ps_2tr_hasync_2thread_Fslice_Tdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_hasync_2thread_Tslice_Fdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_hasync_2thread_Tslice_Fdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'half_async', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': True,
                           'runtime_split_send_recv': True, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': True, 'push_nums': 25}
        self.test_sim_2ps_2tr_hasync_2thread_Tslice_Fdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_hasync_2thread_Fslice_Fdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_hasync_2thread_Fslice_Fdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'half_async', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': True,
                           'runtime_split_send_recv': True, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': True, 'push_nums': 25}
        self.test_sim_2ps_2tr_hasync_2thread_Fslice_Fdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_hasync_2thread_Tslice_Tdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_hasync_2thread_Tslice_Tdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'half_async', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': True,
                           'runtime_split_send_recv': True, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': True, 'push_nums': 25}
        self.test_sim_2ps_2tr_hasync_2thread_Tslice_Tdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_sim_2ps_2tr_hasync_2thread_Fslice_Tdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_hasync_2thread_Fslice_Tdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'half_async', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': True,
                           'runtime_split_send_recv': True, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': True, 'push_nums': 25}
        self.test_sim_2ps_2tr_hasync_2thread_Fslice_Tdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    """sync"""
    def test_sim_2ps_2tr_sync_2thread_Tslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_sync_2thread_Tslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'sync', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': False, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 25}
        self.test_sim_2ps_2tr_sync_2thread_Tslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_sync_2thread_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_sync_2thread_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'sync', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': False, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 25}
        self.test_sim_2ps_2tr_sync_2thread_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_sim_2ps_2tr_sync_2thread_Fslice_Tdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_sync_2thread_Fslice_Tdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'sync', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': False, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 25}
        self.test_sim_2ps_2tr_sync_2thread_Fslice_Tdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_sync_2thread_Tslice_Fdc_Tsm_Fsr_Fgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_sync_2thread_Tslice_Fdc_Tsm_Fsr_Fgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'sync', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': True,
                           'runtime_split_send_recv': False, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 25}
        self.test_sim_2ps_2tr_sync_2thread_Tslice_Fdc_Tsm_Fsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_sim_2ps_2tr_sync_2thread_Fslice_Fdc_Tsm_Fsr_Fgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_sync_2thread_Fslice_Fdc_Tsm_Fsr_Fgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'sync', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': True,
                           'runtime_split_send_recv': False, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 25}
        self.test_sim_2ps_2tr_sync_2thread_Fslice_Fdc_Tsm_Fsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_sim_2ps_2tr_sync_2thread_Tslice_Tdc_Tsm_Fsr_Fgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_sync_2thread_Tslice_Tdc_Tsm_Fsr_Fgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'sync', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': True,
                           'runtime_split_send_recv': False, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 25}
        self.test_sim_2ps_2tr_sync_2thread_Tslice_Tdc_Tsm_Fsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_sync_2thread_Fslice_Tdc_Tsm_Fsr_Fgeo_Twp_Fha_pn25(self):
        """
        test_sim_2ps_2tr_sync_2thread_Fslice_Tdc_Tsm_Fsr_Fgeo_Twp_Fha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'sync', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': True,
                           'runtime_split_send_recv': False, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': False, 'push_nums': 25}
        self.test_sim_2ps_2tr_sync_2thread_Fslice_Tdc_Tsm_Fsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_sync_2thread_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_sync_2thread_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'sync', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': False, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': True, 'push_nums': 25}
        self.test_sim_2ps_2tr_sync_2thread_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_sync_2thread_Fslice_Tdc_Fsm_Fsr_Fgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_sync_2thread_Fslice_Tdc_Fsm_Fsr_Fgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'sync', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': False,
                           'runtime_split_send_recv': False, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': True, 'push_nums': 25}
        self.test_sim_2ps_2tr_sync_2thread_Fslice_Tdc_Fsm_Fsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_sim_2ps_2tr_sync_2thread_Tslice_Fdc_Tsm_Fsr_Fgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_sync_2thread_Tslice_Fdc_Tsm_Fsr_Fgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'sync', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': True,
                           'runtime_split_send_recv': False, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': True, 'push_nums': 25}
        self.test_sim_2ps_2tr_sync_2thread_Tslice_Fdc_Tsm_Fsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_sync_2thread_Fslice_Fdc_Tsm_Fsr_Fgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_sync_2thread_Fslice_Fdc_Tsm_Fsr_Fgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'sync', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': True,
                           'runtime_split_send_recv': False, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': True, 'push_nums': 25}
        self.test_sim_2ps_2tr_sync_2thread_Fslice_Fdc_Tsm_Fsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_sim_2ps_2tr_sync_2thread_Tslice_Tdc_Tsm_Fsr_Fgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_sync_2thread_Tslice_Tdc_Tsm_Fsr_Fgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'sync', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': True,
                           'runtime_split_send_recv': False, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': True, 'push_nums': 25}
        self.test_sim_2ps_2tr_sync_2thread_Tslice_Tdc_Tsm_Fsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)

    def test_sim_2ps_2tr_sync_2thread_Fslice_Tdc_Tsm_Fsr_Fgeo_Twp_Tha_pn25(self):
        """
        test_sim_2ps_2tr_sync_2thread_Fslice_Tdc_Tsm_Fsr_Fgeo_Twp_Tha_pn25
        """
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {'sync_mode': 'sync', 'cpu_num': 2, 'num_threads': 2,
                           'slice_var_up': False, 'enable_dc_asgd': False, 'split_method': True,
                           'runtime_split_send_recv': False, 'geo_sgd': False, 'wait_port': True,
                           'use_hierarchical_allreduce': True, 'push_nums': 25}
        self.test_sim_2ps_2tr_sync_2thread_Fslice_Tdc_Tsm_Fsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
        train_data_list1 = self.get_result(self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)


