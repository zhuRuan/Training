import numpy as np
import pandas as pd
import empyrical as ep

a = pd.DataFrame([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]], columns=['a','b','c','d','e'])
print((a['e']>=4).any())
