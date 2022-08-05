import numpy as np

x = np.random.rand(20, 1)
y = x > 0.5
x = np.random.rand(20, 1)
y_pred = x > 0.5
print(x)
print(np.concatenate((y, y_pred), axis=1))

scene_infos = {
        'Index': [],
        'Start': [],
        'End': []
    }

count = 0
actual = y
for i in range(len(actual) - 1):
    if i == 0:
        if actual[i]:
            count += 1
            scene_infos['Index'].append(count)
            scene_infos['Start'].append(i)
    if not actual[i] and actual[i + 1]:
        count += 1
        scene_infos['Index'].append(count)
        scene_infos['Start'].append(i + 1)
    if actual[i] and not actual[i + 1]:
        scene_infos['End'].append(i + 1)
    if i == len(actual) - 2:
        if actual[i + 1]:
            scene_infos['End'].append(i + 2)

# print(count)
for i in range(count):
    start_actual = scene_infos['Start'][i]
    end_actual = scene_infos['End'][i]

    start_iou = start_actual
    if y_pred[start_actual - 1]:
        for j in range(start_actual - 2, -1, -1):
            if not y_pred[j]:
                start_iou = j + 1
                break

    end_iou = end_actual
    if end_actual != len(y):
        if y_pred[end_actual]:
            for j in range(end_actual + 1, len(actual)):
                if not y_pred[j]:
                    end_iou = j
                    break

    duo_1_count = 0
    for k in range(start_actual, end_actual):
        if y[k] and y_pred[k]:
            duo_1_count += 1

    print(start_iou, end_iou)
    print(duo_1_count)
    print('IOU: ', duo_1_count / (end_iou - start_iou))
