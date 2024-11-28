import matplotlib.pyplot as plt
import numpy as np

img = plt.imread("S1019L01.jpg")

found_circles = [(145, 3, 255), (155, 279, 212), (160, 160, 107), (158, 166, 38)]

canvas = np.zeros_like(img)

for circle in found_circles:
    circle_x, circle_y, circle_radius = circle
    y, x = np.ogrid[-circle_y:img.shape[0]-circle_y, -circle_x:img.shape[1]-circle_x]
    mask = x**2 + y**2 <= circle_radius**2
    canvas[mask] = 0

y, x = np.ogrid[-3:img.shape[0]-3, -145:img.shape[1]-145]
mask = x**2 + y**2 <= 255**2
canvas[mask] = 255

y, x = np.ogrid[-166:img.shape[0]-166, -158:img.shape[1]-158]
mask = x**2 + y**2 <= 38**2
canvas[mask] = 0

y, x = np.ogrid[-160:img.shape[0]-160, -160:img.shape[1]-160]
mask = x**2 + y**2 > 107**2
canvas[mask] = 0

y, x = np.ogrid[-279:img.shape[0]-279, -155:img.shape[1]-155]
mask = x**2 + y**2 > 212**2
canvas[mask] = 0

fig, ax = plt.subplots()

ax.imshow(canvas, cmap='gray')

ax.set_xlim(0, img.shape[1])
ax.set_ylim(0, img.shape[0])
ax.set_aspect('equal', adjustable='box')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Circles with specified regions')
plt.grid(False)
plt.show()
