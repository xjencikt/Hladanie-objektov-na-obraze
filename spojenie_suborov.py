with open('data/moje_zrenicky.txt', 'r') as file1:
    lines_file1 = file1.readlines()

with open('data/jej_zrenicky.txt', 'r') as file2:
    lines_file2 = file2.readlines()

with open('final_mydata.txt', 'a') as file1:
    for i, line in enumerate(lines_file2):
        file1.write(lines_file1[i].strip() + " " + line)

print("Files appended successfully!")
