import matplotlib.pyplot as plt
import json

def print_matrix(plot_path, xlabel, ylabel, x, y, z):
    plt.scatter(x, y, c=z, cmap='viridis')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # https://stackoverflow.com/questions/17682216/scatter-plot-and-color-mapping-in-python
    plt.colorbar()
    plt.savefig(plot_path, dpi=100)

def read_matrix(file_path, plot_path, dataset_sizes, img_sizes):
    with open(file_path, "r") as file_out:
        data = json.loads(file_out.read())
    # print(data)

    accuracys = [round(h["val_acc"],3) for h in data]
    # for ds in dataset_sizes:
    #     for imgs in img_sizes:
    #         for it in data:
    #             if ds == it["dataset_size"] and imgs == it["img_size"]:
    #                 accuracys.append(it["val_acc"])

    """
    [500,1000,1500] => [500,500,500,1000,1000,1000,1500,1500,1500]
    """
    _dataset_sizes = []
    _dataset_sizes = [_dataset_sizes + [size for i in range(len(img_sizes))] for size in dataset_sizes]
    # print(accuracys)

    print_matrix(plot_path, "dataset size", "img size",
                    _dataset_sizes, list(img_sizes)*len(dataset_sizes), accuracys)
