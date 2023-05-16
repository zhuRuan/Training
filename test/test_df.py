import pandas as pd

a = pd.DataFrame(['1','2','3'])
a.index = [1,2,3]
b = pd.DataFrame([True,False,False,False])
print(a[b])