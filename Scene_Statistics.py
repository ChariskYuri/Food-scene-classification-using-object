import numpy as np


def print_scene_statistics(x, class_threshold=0.5, scene_threshold=0.7):
    count = 0
    y = x > class_threshold
    frame_confidence = []
    frame_count = 0
    sus_frame_count = 0
    scene_confidences = []
    scene_infos = []

    # Append related frames
    for i in range(len(y) - 3):
        start = -1
        end = -1
        for j in range(2):
            if x[i + j] >= scene_threshold:
                start = i + j
            if x[i + 3 - j] >= scene_threshold:
                end = i + 3 - j
        if start != -1 and end != -1:
            for j in range(start + 1, end):
                y[j] = True

    # Statistics
    for i in range(len(y) - 1):
        if i == 0:
            if y[i]:
                count += 1
                scene_infos.append(i)
                frame_count += 1
                frame_confidence.append(1)
        if y[i + 1] and not y[i]:
            count += 1
            scene_infos.append(i + 1)
            frame_count += 1
            frame_confidence.append(1)
        if y[i + 1] and y[i]:
            frame_count += 1
            if x[i] < scene_threshold:
                sus_frame_count += 1
                frame_confidence.append(float(x[i + 1]))
        if (not y[i + 1] and y[i]) or (i == len(y) - 2 and y[i + 1]):
            if frame_count == 1:
                scene_confidence = x[i].item()
            else:
                scene_confidence = 1 - ((1 - np.prod(frame_confidence)) * sus_frame_count / frame_count)
            scene_confidences.append(scene_confidence)
            scene_infos.append(frame_count)
            frame_count = 0
            sus_frame_count = 0
            frame_confidence.clear()

    print('Number of scenes: ', count)
    print('Scene confidences:', scene_confidences)
    for i in range(0, len(scene_infos), 2):
        print('Scene {}:'.format(i / 2 + 1),
              'Start: {}'.format(scene_infos[i]),
              'Number of frames: {}'.format(scene_infos[i + 1]))


def print_IOU_score(prediction, actual, class_threshold=0.5):
    prediction = prediction > class_threshold

    IOU_scores = []
    durations = []
    scene_infos = {
        'Index': [],
        'Start': [],
        'End': []
    }

    num_scene = 0
    for i in range(len(actual) - 1):
        if not actual[i] and actual[i + 1]:
            num_scene += 1
            scene_infos['Index'].append(num_scene)
            scene_infos['Start'].append(i + 1)
        if actual[i] and not actual[i + 1]:
            scene_infos['End'].append(i + 1)
        if i == len(actual) - 2:
            if actual[i + 1]:
                scene_infos['End'].append(i + 2)

    for i in range(num_scene):
        start_actual = scene_infos['Start'][i]
        end_actual = scene_infos['End'][i]

        start_iou = start_actual
        if prediction[start_actual - 1]:
            for j in range(start_actual - 2, -1, -1):
                if not prediction[j]:
                    start_iou = j + 1
                    break

        end_iou = end_actual
        if end_actual != len(actual):
            if prediction[end_actual]:
                for j in range(end_actual + 1, len(actual)):
                    if not prediction[j]:
                        end_iou = j
                        break

        duo_1_count = 0
        for k in range(start_actual, end_actual):
            if actual[k] and prediction[k]:
                duo_1_count += 1

        IOU_score = duo_1_count / (end_iou - start_iou)
        durations.append(end_actual - start_actual)
        IOU_scores.append(IOU_score)
        print('Scene {} IOU score: '.format(i + 1), IOU_score)

    np_IOU_scores = np.asanyarray(IOU_scores)
    np_durations = np.asanyarray(durations)
    avg_IOU_score = np.dot(np_IOU_scores, np_durations) / sum(durations)
    print('Average IOU score: ', avg_IOU_score)


def print_metric_scores(prediction, actual, class_threshold=0.5):
    prediction = prediction > class_threshold

    true_positive = np.sum(np.logical_and(actual == 1, prediction == 1))
    true_negative = np.sum(np.logical_and(actual == 0, prediction == 0))
    false_positive = np.sum(np.logical_and(actual == 0, prediction == 1))
    false_negative = np.sum(np.logical_and(actual == 1, prediction == 0))

    precision_1 = true_positive / (true_positive + false_positive)
    recall_1 = true_positive / (true_positive + false_negative)
    f1_score_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1)

    precision_2 = true_negative / (true_negative + false_negative)
    recall_2 = true_negative / (true_negative + false_positive)
    f1_score_2 = 2 * precision_2 * recall_2 / (precision_2 + recall_2)

    print('Food frames:')
    print('\tPrecision:', precision_1)
    print('\tRecall:', recall_1)
    print('\tF1_score:', f1_score_1)

    print('Non-food frames:')
    print('\tPrecision:', precision_2)
    print('\tRecall:', recall_2)
    print('\tF1_score:', f1_score_2)
