import os 

CLASS_NAMES = ["dog", "horse", "elephant", "cat", "cow"]

base_path = "../data/generated_data"
train_files = []
test_files = []

for root, dirs, files in os.walk(base_path):
    animal = root[len(base_path)+1:]
    if len(animal) > 1 and animal in CLASS_NAMES:
        src_gen = os.path.join(base_path, animal)
        des_gen = os.path.join("../data/generated_data", animal)
        num_file = len(files)
        split_num = int(.7*num_file)
        count = 0
        for name in files:
            if count < split_num:
                train_files.append(name)
                count += 1
            else:
                test_files.append(name)