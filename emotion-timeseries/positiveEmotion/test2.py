#-*-coding:utf-8-*-

from torch.utils.data import DataLoader
from test import GetLoader
import numpy as np

# 随机生成数据，大小为10 * 20列
source_data = np.random.rand(10, 20)
# 随机生成标签，大小为10 * 1列
source_label = np.random.randint(0,2,(10, 1))
# 通过GetLoader将数据进行加载，返回Dataset对象，包含data和labels
torch_data = GetLoader(source_data, source_label)

# 读取数据
datas = DataLoader(torch_data, batch_size=6, shuffle=True, drop_last=False, num_workers=2)

for i, data in enumerate(datas):
	# i表示第几个batch， data表示该batch对应的数据，包含data和对应的labels
    d ,label = data
    print(d)
    print(label)
    print("第 {} 个Batch \n{}".format(i, data))