#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author:gentelyang  time:2021-06-15
import paddle
import os

paddle.enable_static()

with open("test_queue_dataset_run_c.txt", "w") as f:
    data = "2 1 2 2 5 4 2 2 7 2 1 3\n"
    data += "2 6 2 2 1 4 2 2 4 2 2 3\n"
    data += "2 5 2 2 9 9 2 2 7 2 1 3\n"
    data += "2 7 2 2 1 9 2 3 7 2 5 3\n"
    f.write(data)
with open("test_queue_dataset_run_d.txt", "w") as f:
    data = "2 1 2 2 5 4 2 2 7 2 1 3\n"
    data += "2 6 2 2 1 4 2 2 4 2 2 3\n"
    data += "2 5 2 2 9 9 2 2 7 2 1 3\n"
    data += "2 7 2 2 1 9 2 3 7 2 5 3\n"
    f.write(data)

slots = ["slot1", "slot2", "slot3", "slot4"]
slots_vars = []
for slot in slots:
    var = paddle.static.data(
        name=slot, shape=[None, 1], dtype="int64", lod_level=1)
    slots_vars.append(var)

dataset = paddle.distributed.QueueDataset()
dataset.init(
    batch_size=1,
    thread_num=2,
    input_type=1,
    pipe_command="cat",
    use_var=slots_vars)
dataset.set_filelist(
    ["test_queue_dataset_run_c.txt", "test_queue_dataset_run_d.txt"])

paddle.enable_static()

place = paddle.CPUPlace()
exe = paddle.static.Executor(place)
startup_program = paddle.static.Program()
main_program = paddle.static.Program()
exe.run(startup_program)

exe.train_from_dataset(main_program, dataset)
with open('./test_queue_dataset_run_c.txt') as f1:
    lines = f1.readlines()
    for line in lines:
        assert (len(line[0])) == 1

with open('./test_queue_dataset_run_d.txt') as f2:
    lines = f2.readlines()
    for line in lines:
        assert (len(line[0])) == 1
os.remove("./test_queue_dataset_run_c.txt")
os.remove("./test_queue_dataset_run_d.txt")