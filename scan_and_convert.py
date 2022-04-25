import os
import subprocess

for cls in os.listdir("."):
    if os.path.isdir(cls) and not cls.__contains__('.'):
        for cls_sub in os.listdir(cls):
            cls_sub_path = os.path.join(cls,cls_sub)
            if os.path.isdir(cls_sub_path) and not cls_sub.__contains__('.'):
                for images in os.listdir(cls_sub_path):
                    image_path = os.path.join(cls_sub_path, images)
                    if not os.path.isdir(image_path) and images.__contains__('best.svg'):
                        command = f"rsvg-convert -b white -h 512 {image_path} > {cls_sub}.png"
                        subprocess.call(command, shell=True)
