import os 
import shutil

translate = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", 
            "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", 
            "mucca": "cow", "pecora": "sheep", "ragno": "spider", "scoiattolo": "squirrel", 
            "dog": "cane", "horse": "cavallo", "elephant" : "elefante", 
            "butterfly": "farfalla", "chicken": "gallina", "cat": "gatto", 
            "cow": "mucca", "spider": "ragno", "squirrel": "scoiattolo", "sheep": "pecora"}

# raw_path = "raw-img/"
# gen_path = "generated_data/"

base_path = "../../cloud_patterns/generated_data"
# destination_path = "../data"

for root, dirs, files in os.walk(base_path):
    animal = root[len(base_path)+1:]
    if len(animal) > 1:
        translated_animal = translate[animal]
        src_gen = os.path.join(base_path, animal)
        src_raw = os.path.join("../../cloud_patterns/raw-img", translated_animal)
        des_gen = os.path.join("../data/generated_data", animal)
        des_raw = os.path.join("../data/raw_data", animal)
        isExist = os.path.exists(des_gen)
        if not isExist:
            os.makedirs(des_gen, exist_ok=False)
        isExist = os.path.exists(des_raw)
        if not isExist:
            os.makedirs(des_raw, exist_ok=False)
        count = 0
        for name in files:
            src_gen_full = os.path.join(src_gen, name)
            src_raw_full = os.path.join(src_raw, name)
            des_gen_full = os.path.join(des_gen, str(count)+"_"+animal+".jpeg")
            des_raw_full = os.path.join(des_raw, str(count)+"_"+animal+".jpeg")
            # print(src_gen_full, des_gen_full)
            # print(src_raw_full, des_raw_full)
            shutil.copyfile(src_gen_full, des_gen_full)
            shutil.copyfile(src_raw_full, des_raw_full)
            count += 1