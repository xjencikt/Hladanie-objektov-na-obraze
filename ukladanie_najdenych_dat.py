import os
import cv2
import numpy as np
import csv


def load_images_from_csv(csv_file, output_file, directory, params):
    prev_x, prev_y, prev_r = None, None, None
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        with open(output_file, 'a') as output:
            for row in csv_reader:
                img_path = os.path.join(directory, row[0])

                if not img_path.endswith(".jpg"):
                    print(f"Warning: Invalid file path '{img_path}'. Skipping...")
                    continue

                img = cv2.imread(img_path)

                if img is None:
                    print(f"Warning: Failed to load image '{img_path}'. Skipping...")
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                denoised_gray = cv2.fastNlMeansDenoising(gray, None, h=31)

                edges = cv2.Canny(denoised_gray, 0, params['upper_limit_canny'])

                circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, params['accumulator_resolution'],
                                           minDist=params['min_center_distance'],
                                           param1=params['upper_limit_canny'],
                                           param2=params['votes_limit'],
                                           minRadius=params['min_circle_size'],
                                           maxRadius=params['min_circle_size'] * 2)

                if circles is not None:
                    image_center = (img.shape[1] // 2, img.shape[0] // 2)
                    distances = [np.linalg.norm((circle[0] - image_center[0], circle[1] - image_center[1])) for circle
                                 in circles[0, :]]

                    closest_circle_index = np.argmin(distances)
                    closest_circle = circles[0, closest_circle_index]

                    x, y, r = int(closest_circle[0]), int(closest_circle[1]), int(closest_circle[2])

                    output.write(f"{x},{y},{r}\n")

                    prev_x, prev_y, prev_r = x, y, r
                else:
                    if prev_x is not None:
                        output.write(f"{prev_x},{prev_y},{prev_r}\n")
                    else:
                        print("Warning: No previous circle information found.")


csv_file = "iris_annotation.csv"
output_file = "data/horne_casti_nespravne.txt"
directory = "duhovky"

params_grid = {
    'accumulator_resolution': [31],
    'votes_limit': [61],
    'upper_limit_canny': [10],
    'min_center_distance': [13],
    'min_circle_size': [176]
}
# Perform grid search
for acc_res in params_grid['accumulator_resolution']:
    for votes_lim in params_grid['votes_limit']:
        for up_lim_canny in params_grid['upper_limit_canny']:
            for min_dist in params_grid['min_center_distance']:
                for min_size in params_grid['min_circle_size']:
                    params = {
                        'accumulator_resolution': acc_res,
                        'votes_limit': votes_lim,
                        'upper_limit_canny': up_lim_canny,
                        'min_center_distance': min_dist,
                        'min_circle_size': min_size
                    }
                    load_images_from_csv(csv_file, output_file, directory, params)
