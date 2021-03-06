import os
from shutil import copyfile, rmtree
from imutils import paths
import random

# https://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
flatten = lambda l: [item for sublist in l for item in sublist]

# https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
def chunkify(lst,n):
    return [lst[i::n] for i in range(n)]

# Division en 2 sous-ensembles "train" et "test" de taille égale
def create_sets(src_folder, dst_folder, size_limit=5000):

    # reset structure
    rmtree(os.path.sep.join([dst_folder, "train"]), ignore_errors=True)
    rmtree(os.path.sep.join([dst_folder, "eval"]), ignore_errors=True)
    os.mkdir(os.path.sep.join([dst_folder, "eval"]))
    os.mkdir(os.path.sep.join([dst_folder, "eval", "cat"]))
    os.mkdir(os.path.sep.join([dst_folder, "eval", "not_cat"]))
    os.mkdir(os.path.sep.join([dst_folder, "train"]))
    os.mkdir(os.path.sep.join([dst_folder, "train", "cat"]))
    os.mkdir(os.path.sep.join([dst_folder, "train", "not_cat"]))

    cat, not_cat = [], []
    for img_path in paths.list_images(src_folder):
        label = img_path.split(os.path.sep)[-2]
        # if label == "cat" and len(cat) < size_limit:
        #     cat.append(img_path)
        # elif label == "not_cat" and len(not_cat) < size_limit:
        #     not_cat.append(img_path)
        # if len(not_cat) >= size_limit and len(cat) >= size_limit:
        #     break
        if label == "cat":
            cat.append(img_path)
        elif label == "not_cat":
            not_cat.append(img_path)

    random.seed(42)
    random.shuffle(cat)
    random.shuffle(not_cat)

    if len(not_cat) > size_limit:
        not_cat = not_cat[0:size_limit]
    if len(cat) > size_limit:
        cat = cat[0:size_limit]

    # get equal ensemble sizes
    if len(not_cat) > len(cat):
        not_cat = not_cat[0:len(cat)]
    elif len(not_cat) < len(cat):
        cat = cat[0:len(not_cat)]

    not_cat_chunks = chunkify(not_cat, 10)
    cat_chunks = chunkify(cat, 10)
    train_set = flatten(not_cat_chunks[:9]) + flatten(cat_chunks[:9])
    eval_set = flatten(not_cat_chunks[9:10]) + flatten(cat_chunks[9:10])
    for img_path in train_set:
        file = img_path.split(os.path.sep)[-1]
        label = img_path.split(os.path.sep)[-2]
        dst = os.path.sep.join([dst_folder, "train", label, file])
        copyfile(img_path, dst)
    for img_path in eval_set:
        file = img_path.split(os.path.sep)[-1]
        label = img_path.split(os.path.sep)[-2]
        dst = os.path.sep.join([dst_folder, "eval", label, file])
        copyfile(img_path, dst)
