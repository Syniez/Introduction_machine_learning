import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
import numpy as np




df = pd.read_excel('./df1.xlsx', engine='openpyxl')
print(df)