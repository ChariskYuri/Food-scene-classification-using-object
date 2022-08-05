import keras
import pandas as pd
from Scene_Statistics import print_scene_statistics, print_IOU_score

# Model, test vid, predictions
model = keras.models.load_model('The_Model_100Fs.h5')

test_data = pd.read_csv('100Fs_Sample_Vids/BE_hHopKqQ8.csv')
test_data.drop(columns=['Unnamed: 0', 'id', 'frame_num'], inplace=True)

test_np_data = test_data.to_numpy()
x_test, y_test = test_np_data[:, :-1], test_np_data[:, -1]

predictions = model.predict(x_test)

# Play around these
classification_threshold = 0.5
scene_threshold = 0.7

# Evaluation
print_scene_statistics(predictions, class_threshold=classification_threshold, scene_threshold=scene_threshold)
print_scene_statistics(y_test, class_threshold=classification_threshold, scene_threshold=scene_threshold)

model.evaluate(x_test, y_test)
print_IOU_score(predictions, y_test, class_threshold=classification_threshold)
