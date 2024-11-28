import os

with open('duhovky/iris_annotation.csv', 'r') as file:
    lines = file.readlines()

with open('data/jej_dolne_casti.txt', 'w') as output_file:
    for line in lines:
        values = line.strip().split(',')
        image_path_parts = values[0].split('/')
        image_filename = image_path_parts[-1]
        image_directory = '/'.join(image_path_parts[:-1])

        if image_filename.endswith('.jpg'):
            image_directory = os.path.join("duhovky", image_directory)

        image_path = os.path.join(image_directory, image_filename)

        if os.path.exists(image_path):
            output_file.write(f"{values[7]},{values[8]},{values[9]}\n")

print("Data has been written to moje_zrenicky.txt successfully.")
