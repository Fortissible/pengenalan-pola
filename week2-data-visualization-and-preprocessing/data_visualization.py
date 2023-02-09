import matplotlib.pyplot as plt
import pandas as pd


def scatter_plots(ax, row, col, data, var_title, data_classes):
    ax[row, col].scatter(data[var_title], data[data_classes])
    ax[row, col].set(xlabel=var_title, ylabel=data_classes)


def line_chart(ax, row, col, data, var_title, data_classes):
    ax[row, col].plot(data[data_classes], label=data_classes)
    ax[row, col].plot(data[var_title], label=var_title)
    ax[row, col].legend(loc="upper left")


def bar_chart(ax, row, col, data, var_title, data_classes):
    ax[row, col].bar(data[var_title], data[data_classes])
    ax[row, col].set(xlabel=var_title, ylabel=data_classes)


def histogram(ax, row, col, data, var_title):
    ax[row, col].hist(data[var_title])
    ax[row, col].set_title(var_title + " histogram")


def subplots(data, type_visual, data_classes):
    var = data.drop(columns=[data_classes])
    var_cnt = len(data.axes[1]) - 1
    fig, ax = plt.subplots(2, var_cnt // 2)
    for idx, var_title in enumerate(var.axes[1]):
        if idx >= var_cnt // 2:
            if type_visual == "scatterplot":
                scatter_plots(ax, 1, idx % (var_cnt // 2), data, var_title, data_classes)
            elif type_visual == "linechart":
                line_chart(ax, 1, idx % (var_cnt // 2), data, var_title, data_classes)
            elif type_visual == "barchart":
                bar_chart(ax, 1, idx % (var_cnt // 2), data, var_title, data_classes)
            elif type_visual == "histogram":
                histogram(ax, 1, idx % (var_cnt // 2), data, var_title)
        else:
            if type_visual == "scatterplot":
                scatter_plots(ax, 0, idx, data, var_title, data_classes)
            elif type_visual == "linechart":
                line_chart(ax, 0, idx, data, var_title, data_classes)
            elif type_visual == "barchart":
                bar_chart(ax, 0, idx, data, var_title, data_classes)
            elif type_visual == "histogram":
                histogram(ax, 0, idx, data, var_title)
    plt.show()


if __name__ == "__main__":
    data_contoh_tipsdb = pd.read_csv("tips.csv")
    print(len(data_contoh_tipsdb.axes[0]), len(data_contoh_tipsdb.axes[1]))
    print(data_contoh_tipsdb.axes[1][0])

    subplots(data_contoh_tipsdb, "scatterplot", "tip")
    subplots(data_contoh_tipsdb, "linechart", "tip")
    subplots(data_contoh_tipsdb, "barchart", "tip")
    subplots(data_contoh_tipsdb, "histogram", "tip")

    print(data_contoh_tipsdb.head(10))
