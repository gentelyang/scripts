#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author:gentelyang  time:2021-06-15
import paddle
import os
paddle.enable_static()

dataset = paddle.distributed.InMemoryDataset()
dataset = paddle.distributed.InMemoryDataset()
slots = ["slot1", "slot2", "slot3", "slot4"]
slots_vars = []
for slot in slots:
    var = paddle.static.data(
        name=slot, shape=[None, 1], dtype="int64", lod_level=1)
    slots_vars.append(var)
dataset.init(
    batch_size=1,
    thread_num=2,
    input_type=1,
    pipe_command="cat",
    use_var=slots_vars)
filelist = ["test_queue_dataset_run_a.txt", "test_queue_dataset_run_b.txt"]
dataset.set_filelist(filelist)
dataset.load_into_memory()
dataset.global_shuffle()
assert dataset.get_shuffle_data_size() == 0
os.remove("./test_queue_dataset_run_a.txt")
os.remove("./test_queue_dataset_run_b.txt")