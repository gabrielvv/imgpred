# IMAGE CLASSIFICATION

```sh
python -c "import keras; print(keras.__version__);"

# activate conda env
activate opencv

# train
python imgpred/train_network.py -d images/train -m models/cat_not_cat_04.model -p reporting/plot_04.png -w 64

# evaluate by hand
python imgpred/bulk_test_network.py -d images/test2 -m models/cat_not_cat_04.model -w 64

# visualize
tensorboard --logdir="./logs"
```

## Python
* http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
* http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
* https://keras.io/getting-started/faq/#what-does-sample-batch-epoch-mean
* https://keras.io/models/model/#fit_generator
* https://keras.io/models/model/#fit

## Datasets

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
| [Caltech101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/) | 9145 |
| [inria](http://lear.inrialpes.fr/~jegou/data.php#holidays) | ?? |
| [ukbench](https://archive.org/details/ukbench) | 2550 |
| **total** | 11694 |

## Résultats

| N°| Res | Training Set Size | Test Set Size | Accuracy | Divers |
| --| ----| ------------------| --------------| ---------| -------|
| 1 | 28 | 637/637* | 144/530| 0.66 ||
| 2 | 28 | 637/626 | 144/530 | 0.75 ||
| 3 | 28 | 637/626 | 144/530 | 0.77 | tensorflow gpu |
| 4 | 64 | 637/626 | 144/530| 0.77 | tensorflow gpu |
| 5 | 28 | 361/361 | 361/361| 0.79 ||
| 6 | 28 | 2295/2295 | 255/255 | 0.87 ||

\* *(avec séries de 4 images similaires)*

## VIZ
<style>img {margin: auto; display: block;}</style>
* Exemples d'images rapportées à la matrice de confusion

![confusion_matrix_samples](./data/viz/conf_matrix_samples.png)

* Progression de la performance au cours de l'apprentissage

![accuracy_loss](./data/viz/plot_05.png)

* Résultats d'un grid search

![plot_gridsearch](./data/viz/plot_gridsearch.png)

* Stats sur le dataset

![img_stats](./data/viz/img_repartition.png)

* Matrice de Confusion après test du classifieur (dataset test2)

![conf_matrix](./data/viz/conf_matrix.png)

* Tensorboard

![tensorboard_scalar](./data/viz/tensorboard_scalar_.png)
![tensorboard_graph](./data/viz/tensorboard_graph_.png)

## TO READ

* https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
* http://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/
* http://cs231n.github.io/neural-networks-3/
* http://cs231n.github.io/understanding-cnn/
* https://hackernoon.com/visualizing-parts-of-convolutional-neural-networks-using-keras-and-cats-5cc01b214e59

## TODO

* entrainement avec et sans **ImageGenerator**
* Supprimer les images qui génèrent le message "iCCP: known incorrect RGB profile"
* Tester test_network par batch de X images (au lieu de 1 à la fois) et évaluer rapidité d'exécution
* OK → Comprendre l'architecture LeNet et les étapes de train_network -> faire des schémas
* Ensembles
  - Pour comparer des classifieurs on utilise l'ensemble **test1**
  - par exemple pour comparer des classifieurs en faisant varier un paramètre du classifieur pour connaitre la meilleure valeur
  - Pour évaluer le classifieur séléctionné on utilise l'ensembe **test2**
* Utiliser d'avantage numpy, sklearn, pandas...
* Packager l'ensemble pour obtenir un projet clés en main
* Afficher les valeurs finales obtenues sur les graphes + max/min
* Créer un fichier rapport exhaustif de chaque essai (train+test) + images de résultats
* A quoi sert le **ImageDataGenerator** → image data preparation and augmentation?
* Que représente **loss** → fonction d'évaluation du réseau lors de la phase d'entrainement -> binary cross entropy ?
* Que représente **epoch** et **batch** ?
    * **Batch**: a set of N samples. The samples in a batch are processed independently, in parallel. If training, a batch results in only one update to the model. The batch size in iterative gradient descent is the number of patterns shown to the network before the weights are updated. It is also an optimization in the training of the network, defining how many patterns to read at a time and keep in memory.
    * **Epoch**: an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation. The number of epochs is the number of times that the entire training dataset is shown to the network during training.
* Pourquoi LeNet ne se stabilise pas autour de la meilleure performance observée lors des itérations?
* Expliquer:
  * pourquoi la perf annoncée du classifieur après la phase d'entraînement peut différer de celle obtenue lors de la phase d'évaluation
    * variété des images
  * pourquoi avec les mêmes paramètres d'entrainement on obtient des classifieurs plus ou moins performants?
    * différences dans les données d'entrainement
* Comparer les résultats avec différentes tailles / réseaux
* confronter à des images proches du chat (chiens, autres félins, "cougar_face" dans Object101 dataset)
* Use a deeper network architecture during training.
