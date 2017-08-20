import numpy as np

aa = [1, 2, 3]
a1 = np.argmax(aa)
print a1
bb = [aa]
a2 = np.argmax(bb[0])
print aa.index(2)