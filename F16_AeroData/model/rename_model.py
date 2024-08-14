import os
import csv


path = "./"
dirs = os.listdir(path)
tmp = open("./model_name.csv", 'w', newline='')
csv_write = csv.writer(tmp)
csv_write.writerow(["name", "test_r2", "test_error"])
for i in dirs:
    if os.path.splitext(i)[1] == ".pth":
        raw_name = os.path.splitext(i)[0].split("-")
        if len(raw_name) == 3:
            csv_write.writerow([raw_name[0], raw_name[1], raw_name[2]])
        else:
            csv_write.writerow([raw_name[0], raw_name[1], raw_name[2] + "-" + raw_name[3]])
        os.rename(path + i, path + raw_name[0] + ".pth")
tmp.close()
