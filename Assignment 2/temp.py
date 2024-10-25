import numpy as np
from oracles_updated import f1, f2, f3
sr_no = 22585
boolian = True
vec = np.array([1,2,3,4,5])
temp = 0

a,b = f1(sr_no, boolian)
c = f2(vec, sr_no, temp)
d = f3(vec, sr_no, temp)

print(a)
print(b)
print(c)
print(d)