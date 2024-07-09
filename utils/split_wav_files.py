import glob
import os
import math

files = glob.glob("./train_wav_files/*")
num_split = 5
files_per_split = int(math.ceil(len(files) / 5))

for j in range(num_split): 
    os.mkdir(f"./train_wav_files/split_{j}")

for round in range(num_split): 
    for i in range(files_per_split): 
        if (round * num_split + i < len(files)): 
            cur_file = files[round * files_per_split + i]
            print("".join(cur_file.split("\\")[:-1]) + f"/split_{round}/" + cur_file.split("\\")[-1])
            os.rename(cur_file.replace("\\", "/"), "".join(cur_file.split("\\")[:-1]) + f"/split_{round}/" + cur_file.split("\\")[-1])

