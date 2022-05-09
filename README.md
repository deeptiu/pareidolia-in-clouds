# pareidolia-in-clouds

In this project "Pareidolia in Clouds" we aim to model pattern abstractions that cannot be inherently modeled by mathematical equations. By focusing on learning patterns in clouds we aim to learn latent pattern representations in a creative fashion. We will be using CLIPasso as our backbone which will enable us to abstract sketch like representations of the cloud images. We will be focusing on finding animal images within the clouds. One of the parts we explored in depth was generating cloud like images of animals since there does not exist an extensive cloud database to train on. Our results show that it is possible to learn patterns in clouds that might not be obvious. 

You can find our full project details here https://sites.google.com/andrew.cmu.edu/pareidoliainclouds

Steps to obtain the classification results.

1. Clone the repo: 
```bash
git clone https://github.com/deeptiu/pareidolia-in-clouds.git
```
2. Download the Animals-10 dataset from Kaggle https://www.kaggle.com/datasets/alessiocorrado99/animals10 and place under src/. This folder should be named `raw-img`.
3. Run the script to create style transferred images:
```bash
cd src
python mask_rcnn_style_transfer.py
```
4. Collect all the CLipasso output images to this folder 
5. Run the script to convert svg images to pngs 
```bash
sh scan_and_convert.py 
```
6. For training copy the images to `src/images` and for testing copy the images to `src/test_images`
7. Run the classifier to train/test 
```bash
python src/classifier.py 
```
