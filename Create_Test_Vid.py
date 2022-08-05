import pandas as pd
import numpy as np
import Scene_Statistics
from sklearn.feature_selection import SelectKBest
df = pd.read_csv('Sample_Labeled_AIP_Data.csv', index_col=False)
df = df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.2', '_id', 'picName', ''])

print(df.columns)
id = 'Fv6MLA-f9L8'
id_df = df[df['id'] == id].sort_values(by=['frame_num'], ascending=True)


id_df.to_csv('Sample_Vids/{}.csv'.format(id))