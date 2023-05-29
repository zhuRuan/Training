# -*- coding: utf-8 -*-

import pandas as pd

def read_China_Securities_index(file_path='../is/index/中证500行情.xlsx'):
    _China_index = pd.read_excel(file_path)
    print(_China_index)

read_China_Securities_index()
