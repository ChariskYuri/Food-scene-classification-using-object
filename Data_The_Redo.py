import pandas as pd
import numpy as np
import Scene_Statistics

# # Get a test video.
# df = pd.read_csv('New_AIP_Data.csv', index_col=False, usecols=['id', 'frame_num'])
# id = 'Ep7pKw4mLiM'
# a_vid = df[df['id'] == id].sort_values(by='frame_num', ascending=True)
#
# a_vid.to_csv('Test_vids.csv')

# # Split a gigantic csv file into smaller 20000-ish lines csv files.
# source_path = "New_AIP_Data.csv"
# for i, chunk in enumerate(pd.read_csv(source_path, chunksize=20000)):
#     chunk.to_csv('splited_new/chunk{}.csv'.format(i), index=False)

# # Label the data.
# df = pd.read_csv('New_AIP_Data.csv', index_col=False)
# print(df.columns)
# anno = pd.read_csv('New_Annotation.csv', index_col=False)
# np_anno = anno.to_numpy()
# np_df = df.to_numpy()

# food_labels = []
# for i in range(len(np_df)):
#     food = False
#     for j in range(len(np_anno)):
#         if np_df[i][1001] == np_anno[j][1]:
#             if np_anno[j][2] <= np_df[i][1002] <= np_anno[j][3]:
#                 food = True
#                 continue
#     if food:
#         food_labels.append(1)
#     else:
#         food_labels.append(0)
#
# df['food'] = food_labels
# print(df['food'].value_counts())
#
# df.to_csv('Labeled_AIP_Data.csv')

# # Get feed data.
# df = pd.read_csv('Labeled_AIP_Data.csv', index_col=False)
#
# dropped_df = df.drop(columns=['picName', 'id', 'frame_num', 'Unnamed: 0.1', 'Unnamed: 0', '_id'], inplace=False)
#
# dropped_df.to_csv('Feed_AIP_Data.csv')

# Get a sample of a .csv file
df = pd.read_csv('New_AIP_Data.csv', index_col=False)
print(df.columns)
sample_df = df.head(20)
sample_df.to_csv('Sample_New_AIP_Data.csv', index=False)
