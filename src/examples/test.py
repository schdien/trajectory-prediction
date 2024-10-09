import numpy as np
import time

a = np.random.rand(500,2)
b = np.random.rand(600,200,2)

time_start = time.time()  # 记录开始时间

a_pad = a[np.newaxis:,np.newaxis,:]
b_pad = b[:,np.newaxis,:,:]
distances = np.linalg.norm(a_pad-b_pad, axis=-1)

time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print(time_sum)