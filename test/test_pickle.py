import os.path
import pickle
v = {'str33324':'222'}
v['GG'] = 'GG'
v['aa'] = {}
v['aa']['cc'] = 'bb'
if 'aa' in v.keys():
    v['aa']['dd'] = 'cc'
python_variable_path = 'D:\\Ruiwen\\PythonProject\\Training\\test\\python_variable.pkl'
if os.path.exists(python_variable_path):
    old_plot_dict_dict = []
    with open(python_variable_path, 'rb') as f:
        old_plot_dict_dict = pickle.load(f)
    old_plot_dict_dict.update(v)
    print(old_plot_dict_dict)
    with open(python_variable_path, 'wb') as f2:
        pickle.dump(old_plot_dict_dict, f2, 0)
else:
    with open(python_variable_path, 'wb') as f2:
        pickle.dump(v, f2, 0)