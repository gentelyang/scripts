#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author:gentelyang  time:2021-04-19
from paddle.fluid.core import GraphPyService,GraphPyServer,GraphPyClient
import numpy as np
ips_str = "127.0.0.1:4211;127.0.0.1:4212"  #打算启动2台server，ip分别是127.0.0.1:4211127.0.0.1:4212
server1 = GraphPyServer()
server2 = GraphPyServer()
client1 = GraphPyClient()
client2 = GraphPyClient() #模拟分布式环境，启动2台client

node_types = ["user"] #节点table名
edge_types = ["user2item"]  #边table名

server1.set_up(ips_str,127,node_types,edge_types,0) #启动server1，需要5个参数，ip列表，shard个数，节点tashard个数，节点table名数组，边table名数组，server编号（下面的client的set-up和这个用法类似，最后一个参数表示client编号）
server1.add_table_feat_conf("user", "a", "float32", 1) #给user表添加一个属性a，类型为float32，个数为1
server1.add_table_feat_conf("user", "b", "int32", 2)
server1.add_table_feat_conf("user", "c", "string", 1)
server1.add_table_feat_conf("user", "d", "float32", 1)

server2.set_up(ips_str,127,node_types,edge_types,1)
server2.add_table_feat_conf("user", "a", "float32", 1)
server2.add_table_feat_conf("user", "b", "int32", 2)
server2.add_table_feat_conf("user", "c", "string", 1)
server2.add_table_feat_conf("user", "d", "float32", 1)

client1.set_up(ips_str,127,node_types,edge_types,0)
client2.set_up(ips_str,127,node_types,edge_types,1)
server1.start_server(False) #启动server，参数=False表示不阻塞，代码继续执行，如果参数=True则阻塞，直到client发送stop-server命令才会解除阻塞。
server2.start_server(False)
print("server start success ...")
client1.start_client()
client2.start_client()
print("client start success ...")

client1.bind_local_server(0,server1) #标记0号channel为本地channel，并绑定本地server，这个函数是优化本地查询的，实际测试过程中可以不用）
client2.bind_local_server(1,server2)#同上
print("client bind_local_server success ...")

client1.load_edge_file("user2item", "input.txt", 0) #导入边文件
client1.load_node_file("user", "node_input.txt") #导入点文件
list = client2.pull_graph_list("user2item",0,1,4,1) #批量查询节点，table名为user2item，server编号为0，节点id为1，从第4个元素开始遍历，步长为1
for x in list:
    print("pull_graph_list", x)

list = client1.batch_sample_neighboors("user2item",[96], 4) #从user2item表里采样，采样节点为96，返回4个96号节点的邻居
for x in list:
    print("batch_sample_neighboors", x)
list = client1.random_sample_nodes("user",0,6) #从0号server的user表里随机返回6个node
print("sample nodes result")
for x in list:
    print("random_sample_nodes", x)

# string
print("test string")
print(client1.get_node_feat("user", [37, 96], ["c"])) #从user表里，把[37,96]号节点取出，然后获取其属性c

# int
print("test int")
data = client1.get_node_feat("user", [37, 96], ["b"])
print(np.frombuffer(data[0][0], "int32"))

# float
print("test float")
data = client1.get_node_feat("user", [37, 96], ["a", "d"])
print(np.frombuffer(data[0][0], "float32"))