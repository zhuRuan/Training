from read_data.generate_random_data import CAP_matrix

matrix = CAP_matrix(50,60)
rank_matrix = matrix.rank(axis=1, method='first')
# `print(rank_matrix)`