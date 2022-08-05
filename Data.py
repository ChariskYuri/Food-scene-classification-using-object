import pandas as pd
import imblearn as imb
from collections import Counter
import numpy as np
# Split a gigantic csv file into smaller 20000-ish lines csv files.
# source_path = "AIP_Data.csv"
# for i, chunk in enumerate(pd.read_csv(source_path, chunksize=20000)):
#     chunk.to_csv('splited/chunk{}.csv'.format(i), index=False)

df = pd.read_csv('200Fs_Feed_AIP_Data.csv', index_col=False)
df.drop(columns=['Unnamed: 0'], inplace=True)

column_list = list(df.columns)

np_data = df.to_numpy()
x, y = np_data[:, :-1], np_data[:, -1]

counter = Counter
print(Counter(y))  # {0.0: 205124, 1.0: 105952}

over = imb.over_sampling.SMOTE(sampling_strategy=1)
steps = [('o', over)]
pipeline = imb.pipeline.Pipeline(steps=steps)
x, y = pipeline.fit_resample(x, y)

print(Counter(y))
y = np.reshape(y, newshape=(410248, 1))
#
# print(x.shape, y.shape)
# print(Counter(y))
#
data = np.concatenate((x, y), axis=1)
print(data.shape)
SMOTE_df = pd.DataFrame(data, columns=column_list)

SMOTE_df.to_csv('SMOTE_200Fs_Feed_AIP_Data.csv')









