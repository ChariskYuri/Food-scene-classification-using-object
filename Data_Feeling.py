import pandas as pd
import numpy as np

df = pd.read_csv('Labeled_AIP_Data.csv', index_col=False)
df = df.drop(columns=['picName', 'id', 'frame_num', 'Unnamed: 0.1', 'Unnamed: 0', '_id'],
             inplace=False)

food_mean_df = df[df['food'] == 1].mean()
non_food_mean_df = df[df['food'] == 0].mean()

sub_mean_df = food_mean_df.subtract(non_food_mean_df)
sub_mean_df_top = sub_mean_df.sort_values(ascending=False).head(21)
sub_mean_df_bot = sub_mean_df.sort_values(ascending=True).head(21)

print(sub_mean_df)
sub_mean_df_top.to_csv('Subtracted mean top.csv')
sub_mean_df_bot.to_csv('Subtracted mean bot.csv')

