import matplotlib.pyplot as plt
import numpy as np

# SEE https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.annotate
def print_save_plot(N, H, plot_path):
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")

    final_val_loss = H.history["val_loss"][-1]
    final_loss = H.history["loss"][-1]
    final_acc = H.history["acc"][-1]
    final_val_acc = H.history["val_acc"][-1]

    # https://matplotlib.org/gallery/pyplots/pyplot_annotate.html#sphx-glr-gallery-pyplots-pyplot-annotate-py
    def draw_arrow(x, y, up_or_down, prepend_text=""):
        # plt.annotate(str(v), xy=(N-1, v))
        # ATTENTION il faut s'adapter à l'intervalle de valeurs représentée sur chaque axe
        # l'offset sera visuellement plus ou moins important
        h_offset = x + 1
        v_offset = y+(+0.05 if up_or_down == "up" else -0.05)
        plt.annotate(prepend_text + "\n" + str(y), xy=(x, y), xytext=(h_offset, v_offset),
            arrowprops=dict(facecolor='black', shrink=0.05),
        )

    draw_arrow(N-1, final_val_loss, "up" if final_val_loss > final_loss else "down", "val_loss:")
    draw_arrow(N-1, final_loss, "up" if final_loss > final_val_loss else "down", "loss:")
    draw_arrow(N-1, final_acc, "up" if final_acc > final_val_acc else "down", "acc:")
    draw_arrow(N-1, final_val_acc, "up" if final_val_acc > final_acc else "down", "val_acc:")

    plt.title("Training Loss and Accuracy on Cat/Not Cat")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plot_path, dpi=100)
