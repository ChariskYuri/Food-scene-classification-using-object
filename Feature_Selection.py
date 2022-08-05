from sklearn.feature_selection import SelectKBest
import pandas as pd
import numpy as np

df = pd.read_csv('Feed_AIP_Data.csv', index_col=False)
df.drop(columns=['Unnamed: 0'], inplace=True)
print(df.columns)

column_list = list(df.columns)
np_data = df.to_numpy()
x, y = np_data[:, :-1], np_data[:, -1]

selector = SelectKBest(k=100).fit(x, y)
X_new = selector.transform(x)
new_columns = selector.get_feature_names_out(input_features=column_list[:-1])
new_columns = np.append(new_columns, ['food', 'id', 'frame_num'])
print(new_columns)

df = pd.read_csv('Labeled_AIP_Data.csv', index_col=False, usecols=new_columns)
print(df.columns)

id = 'BE_hHopKqQ8'
id_df = df[df['id'] == id].sort_values(by=['frame_num'], ascending=True)


id_df.to_csv('100Fs_Sample_Vids/{}.csv'.format(id))
# y = np.reshape(y, newshape=(311076, 1))
#
# print(X_new.shape, y.shape)
# # print(Counter(y))
#
# data = np.concatenate((X_new, y), axis=1)
# print(data.shape)
# SMOTE_df = pd.DataFrame(data, columns=new_columns)
#
# SMOTE_df.to_csv('200Fs_Feed_AIP_Data.csv')