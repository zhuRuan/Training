from read_data.generate_random_data import CAP_matrix
import pandas as pd
import numpy as np

a = pd.Series([1,np.nan,np.nan,4,np.nan,6,0])
a.replace(0,np.nan,inplace=True)

matrix = CAP_matrix(50,60)
rank_matrix = matrix.rank(method='first')
rank_matrix2 = a.rank(method='dense', ascending=True, pct=True)
print(rank_matrix2)

