import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report

def show_histogram(data, title="Histogram", bins=20, hist_color="green"):
    plt.figure(figsize=(10, 7))
    plt.hist(data,bins,color=hist_color)
    plt.title(title)
    plt.ylabel("Count")

    plt.show()

def show_boxplot(iris_data_noclass):
    plt.figure(figsize=(10, 7))
    iris_data_noclass.boxplot()
    plt.show()

def show_scatter_plot(data,data_class):
    var_cnt = len(data.axes[1])-1
    fig, ax = plt.subplots(2, var_cnt // 2)
    for idx, var_title in enumerate(data.axes[1]):
        if idx <= var_cnt-1:
            ax[idx//2, idx%2].scatter(data[data_class], data[var_title])
            ax[idx//2, idx%2].set(xlabel=data_class, ylabel=var_title)
    plt.show()

def naive_bayes_gaussian(iris_data_noclass, iris_data_class):
    # Split data uji dan data latih dengan proporsi 1:3
    x_train, x_test, y_train, y_test = train_test_split(
        iris_data_noclass, iris_data_class,test_size=0.25, random_state=5)

    # Pakai fungsi naive bayes gausian dari pkg scikit
    iris_naivebayes_model = GaussianNB()

    # Input data latih (fitur dan kelas) untuk latih model dengan fungsi fit
    naivebayes_train = iris_naivebayes_model.fit(x_train, y_train)

    # Gunakan model naive bayes gausian yg telah dilatih utk prediksi kelas data uji
    iris_data_class_pred = naivebayes_train.predict(x_test)

    # Measurement
    df_hasil_prediksi = pd.DataFrame({"Prediksi": iris_data_class_pred, "KelasIris": y_test})
    print(df_hasil_prediksi, "\n")
    print("probabilitas prediksi", "\n", naivebayes_train.predict_proba(x_test), "\n")
    print("Confussion Matrix", "\n", confusion_matrix(y_test, iris_data_class_pred), "\n")
    print("Report Klasifikasi", "\n", classification_report(y_test, iris_data_class_pred), "\n")

if __name__ == "__main__":
    iris_data = pd.read_csv("iris.csv")
    print(iris_data.head(15),"\n")
    print(iris_data.describe())

    #Persiapan dataset iris
    iris_data_class = iris_data["variety"]
    print(iris_data_class.head(), "\n")
    iris_data_noclass = iris_data.drop(columns=["variety"])
    print(iris_data_noclass.head(), "\n")

    #Tampilkan visualisasi sebaran data
    show_histogram(data=iris_data["sepal.length"],title="Sepal length in cm")
    show_histogram(data=iris_data["sepal.width"], title="Sepal width in cm")
    show_histogram(data=iris_data["petal.length"], title="Petal length in cm")
    show_histogram(data=iris_data["petal.width"], title="Petal width in cm")
    show_boxplot(iris_data_noclass)
    show_scatter_plot(data=iris_data,data_class="variety")

    #Naive Bayes
    naive_bayes_gaussian(iris_data_noclass, iris_data_class)


