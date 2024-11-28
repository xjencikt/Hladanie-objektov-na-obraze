import os
import cv2
import numpy as np


def load_images_from_directory(directory, params):
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith((".jpg", ".png", ".jpeg")):
                filepath = os.path.join(root, filename)

                img = cv2.imread(filepath)

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                denoised_gray = cv2.fastNlMeansDenoising(gray, None, h=0)

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

                    # Find the index of the circle closest to the image center and lower than the bottom
                    # valid_circles = [(circle[0], circle[1], circle[2]) for circle in circles[0, :]
                    #                  if circle[1] > image_center[1]]
                    # if valid_circles:
                    #     closest_circle = min(valid_circles, key=lambda x: np.linalg.norm(
                    #         (x[0] - image_center[0], x[1] - image_center[1])))
                    #     center_x = closest_circle[0]
                    #     center_y = closest_circle[1]

                    valid_circles = [(circle[0], circle[1], circle[2]) for circle in circles[0, :]
                                     if circle[1] < image_center[1]]
                    if valid_circles:
                        closest_circle = min(valid_circles, key=lambda x: np.linalg.norm(
                            (x[0] - image_center[0], x[1] - image_center[1])))
                        center_x = closest_circle[0]
                        center_y = closest_circle[1]

                        # Adjust the x-coordinate of the center to be close to the image center
                        if center_x < image_center[0] - 100:
                            center_x += 100
                        elif center_x > image_center[0] + 100:
                            center_x -= 100

                        center = (int(center_x), int(center_y))
                        radius = int(closest_circle[2])
                        cv2.circle(img, center, radius, (0, 255, 0), 2)

                cv2.imshow('Circles Detected', img)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()


directory = "duhovky"

params_grid = {
    'accumulator_resolution': [30],
    'votes_limit': [94],
    'upper_limit_canny': [67],
    'min_center_distance': [12],
    'min_circle_size': [183]
}

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
                    load_images_from_directory(directory, params)
