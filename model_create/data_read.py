import pandas as pd
X = pd.read_csv('../../parameters_250000.txt', sep=' ')
Y = pd.read_csv('../../results.txt', sep=' ', index_col=False)
# These functional groups do not exist in my model
Y = Y.drop(['light_aromatic_C-C', 'light_aromatic_methoxyl'], axis=1)
y_columns = Y.columns.values
X = X.values.astype(np.float32)
Y = Y.values.astype(np.float32)
print(X)
print(Y)
