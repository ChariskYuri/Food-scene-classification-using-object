import keras
import numpy as np
import tensorflow as tf
import pandas as pd
from Scene_Statistics import print_scene_statistics

df = pd.read_csv('SMOTE_Feed_AIP_Data.csv', index_col=False)
df.drop(columns=['Unnamed: 0'], inplace=True)

# food_data = df[df['food'] == 1]
# non_food_data = df[df['food'] == 0]
#
# food_np_data = food_data.to_numpy()
# non_food_np_data = non_food_data.to_numpy()
# train_data = np.concatenate((food_np_data, non_food_np_data), axis=0)
train_data = df.to_numpy()
np.random.shuffle(train_data)

x, y = train_data[:, :-1], train_data[:, -1]
print(x.shape, y.shape)
model = keras.Sequential([
    # keras.layers.Dense(units=500, activation='relu'),
    keras.layers.Dense(units=100, activation='relu'),
    keras.layers.Dense(units=50, activation='relu'),
    keras.layers.Dense(units=1, activation='sigmoid'),
])

callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'], callbacks=[callback])

model.fit(x=x, y=y, epochs=50, verbose=2, validation_split=0.2, batch_size=32)

# test_data = pd.read_csv('Sample_Vids/BE_hHopKqQ8.csv')
# test_data.drop(columns=['Unnamed: 0', '_id', 'picName', 'id', 'frame_num'], inplace=True)
#
# test_np_data = test_data.to_numpy()
# x_test, y_test = test_np_data[:, :-1], test_np_data[:, -1]
#
# predictions = model.predict(x_test)
#
# print_scene_statistics(predictions, scene_threshold=0.7)
# print_scene_statistics(y_test, scene_threshold=0.7)
#
# print(model.evaluate(x_test, y_test))

# model.save('The_Model_200Fs.h5')
