# IMAGE CLASSIFICATION

```sh
python train_network.py -d images/train -m models/cat_not_cat_04.model -p reporting/plot_04.png -w 64

python bulk_test_network.py -d images/test -m models/cat_not_cat_04.model -w 64
```

## Dataset

Cat:
* [google image](https://www.google.com/search?q=cat) → 610
* [imagenet](www.image-net.org/synset?wnid=n02121620) → 861
* total → 1471

Not Cat:
* [ukbench](https://archive.org/details/ukbench) -> 2550

## Pistes d'amélioration

* Gather additional training data (ideally, 5,000+ example "Cat" images).
* Utilize higher resolution images during training. I imagine 64×64 pixels would produce higher accuracy. 128×128 pixels would likely be ideal (although I have not tried this).
* Use a deeper network architecture during training.

## 1er classifieur

IMAGE RESOLUTION: 28x28

TRAINING:
* cat images → 637  
* not_cat images → 637 **(avec séries de 4 images similaires)**

TEST:
* cat images → 144
* not_cat images → 530

ACCURACY: **0.66**  

## 2eme classifieur

IMAGE RESOLUTION: 28x28

TRAINING:
* cat images → 637  
* not_cat images → 626

TEST:
* cat images → 144
* not_cat images → 530

ACCURACY: **0.75**

## 3eme classifieur

TENSORFLOW GPU
IMAGE RESOLUTION: 28x28
TRAINING:
* cat images → 637  
* not_cat images → 626
TEST:
* cat images → 144
* not_cat images → 530
ACCURACY: **0.77**

## 4eme classifieur

TENSORFLOW GPU
IMAGE RESOLUTION: 64x64
TRAINING:
* cat images → 637  
* not_cat images → 626
TEST:
* cat images → 144
* not_cat images → 530
tp=60 tn=461 fp=69 fn=84
ACCURACY: **0.77**

## 5eme classifieur
