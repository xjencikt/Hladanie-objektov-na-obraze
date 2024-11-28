import math

def calculate_iou(circle1, circle2):
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2

    distance_centers = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    if distance_centers >= r1 + r2:
        return 0.0

    if distance_centers <= abs(r1 - r2):
        return min(math.pi * r1**2, math.pi * r2**2) / max(math.pi * r1**2, math.pi * r2**2)

    intersection_area = r1**2 * math.acos((distance_centers**2 + r1**2 - r2**2) / (2 * distance_centers * r1)) \
                         + r2**2 * math.acos((distance_centers**2 + r2**2 - r1**2) / (2 * distance_centers * r2)) \
                         - 0.5 * math.sqrt((-distance_centers + r1 + r2) * (distance_centers + r1 - r2) \
                                           * (distance_centers - r1 + r2) * (distance_centers + r1 + r2))

    union_area = math.pi * r1**2 + math.pi * r2**2 - intersection_area

    iou = intersection_area / union_area

    return iou

def read_circle_file(file_path):
    circles = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y, r = map(int, line.strip().split(','))
            circle = (x, y, r)
            circles.append(circle)
    return circles


def evaluate_detection(found_circles, true_circles, threshold=0.75):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for found_circle in found_circles:
        iou_max = 0
        for true_circle in true_circles:
            iou = calculate_iou(found_circle, true_circle)
            if iou > iou_max:
                iou_max = iou

        if iou_max >= threshold:
            true_positives += 1
        else:
            false_positives += 1

    for true_circle in true_circles:
        found = False
        for found_circle in found_circles:
            iou = calculate_iou(found_circle, true_circle)
            if iou >= threshold:
                found = True
                break
        if not found:
            false_negatives += 1

    return true_positives, false_positives, false_negatives


found_circle_file = "data/horne_casti_nespravne.txt"
true_circle_file = "data/jej_horne_casti.txt"

found_circles = read_circle_file(found_circle_file)
true_circles = read_circle_file(true_circle_file)

true_positives, false_positives, false_negatives = evaluate_detection(found_circles, true_circles)
print("True Positives:", true_positives)
print("False Positives:", false_positives)
print("False Negatives:", false_negatives)

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)