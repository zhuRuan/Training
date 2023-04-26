from read_data.generate_random_data import CAP_matrix
import pandas as pd
import numpy as np

a = pd.Series([1,2,3,4,5,6,0])
a.replace(0,np.nan,inplace=True)

matrix = CAP_matrix(50,60)
rank_matrix = matrix.rank(axis=1, method='first')
rank_matrix2 = matrix.rank(axis=1,method='dense', pct=True)
print(rank_matrix)

