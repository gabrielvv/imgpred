# IMAGE CLASSIFICATION

```sh
python -c "import keras; print(keras.__version__);"

# activate conda env
activate opencv

# train
python train_network.py -d images/train -m models/cat_not_cat_04.model -p reporting/plot_04.png -w 64

# evaluate by hand
python bulk_test_network.py -d images/test2 -m models/cat_not_cat_04.model -w 64

# visualize
tensorboard --logdir="./logs"
```

## Python links
* http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
* http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
* https://keras.io/getting-started/faq/#what-does-sample-batch-epoch-mean
* https://keras.io/models/model/#fit_generator
* https://keras.io/models/model/#fit

## Dataset

#### Cat

| Source | Total |
| --------------| -----------|
| [google image](https://www.google.com/search?q=cat&source=lnms&tbm=isch&sa=X&ved=0ahUKEwi8oKaNw5jYAhWG8RQKHVfoD7kQ_AUICigB&biw=1920&bih=968) | 610 |
| [imagenet](https://www.image-net.org/synset?wnid=n02121620) | 861 |
| [kaggle](https://www.kaggle.com/c/dogs-vs-cats/data) | 12500 |
| **total** | 13471 |

#### !Cat

| Source | Total |
| --------------| -----------|
| [Caltech101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/) | ?? |
| [inria](http://lear.inrialpes.fr/~jegou/data.php#holidays) | ?? |
| [ukbench](https://archive.org/details/ukbench) | 2550 |

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

IMAGE RESOLUTION: 28x28  

TRAINING:
* cat images → 361  
* not_cat images → 361  

TEST:
* cat images → 361
* not_cat images → 361  

ACCURACY: **0.79**

## 6eme classifieur

IMAGE RESOLUTION: 28x28  
TRAINING:
* cat images → 2295  
* not_cat images → 2295  

TEST:
* cat images → 255
* not_cat images → 255  

ACCURACY: **0.87**

## TO READ

* https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
* http://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/
* http://cs231n.github.io/neural-networks-3/
* http://cs231n.github.io/understanding-cnn/
* https://hackernoon.com/visualizing-parts-of-convolutional-neural-networks-using-keras-and-cats-5cc01b214e59
