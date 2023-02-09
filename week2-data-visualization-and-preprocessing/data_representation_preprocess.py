import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def importImage(imageName,title="Image"):
    data = cv2.imread(imageName)
    cv2.imshow(title, data)
    cv2.waitKey(0)
    return data


def line_chart(data, data_classes):
    fig, (ax) = plt.subplots(1, 1)
    ax.plot(data[data_classes], label=data_classes)
    ax.legend(loc="upper left")
    plt.show()

def numeric_preprocessing(table):
    table.columns = ["times", "data_sent"]

    mean_data_send = np.mean(table["data_sent"])
    table.data_sent[np.isnan(table["data_sent"])] = mean_data_send
    print(table.head(10), "\n")

    table["data_sent"] = (table["data_sent"] - table["data_sent"].min()) / (
                table["data_sent"].max() -table["data_sent"].min())

    print(table.head(10))
    line_chart(table, "data_sent")

def img_hist_eq(img):
    R, G, B = cv2.split(img)
    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)
    img_equalized = cv2.merge((output1_R, output1_G, output1_B))

    res = np.hstack((img, img_equalized))

    cv2.imshow("before-after equalized", res)
    cv2.waitKey(0)

    babon_hist = cv2.calcHist([img],[0],None,[256],[0,256])
    babon_equ_hist = cv2.calcHist([img_equalized], [0], None, [256], [0, 256])

    fig, (ax1,ax2) = plt.subplots(1, 2)
    ax1.plot(babon_hist, label="pixel frequencies")
    ax1.set_title("babon histogram")
    ax1.legend(loc="upper left")
    ax2.plot(babon_equ_hist, label="pixel frequencies")
    ax2.set_title("babon equalized histogram")
    ax2.legend(loc="upper left")
    plt.show()

def img_bg_segmentation(img):
    R, G, B = cv2.split(img)
    images = [R, G, B]
    titles = ["red", "green", "blue"]
    for i in range(3):
        plt.subplot(1, 3, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

    ret, mask = cv2.threshold(R, 240, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    resultR = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=10)

    image_segmented_from_bg = cv2.bitwise_and(img, img, mask=resultR)
    cv2.imshow("segmented", image_segmented_from_bg)
    cv2.waitKey(0)

if __name__ == "__main__":
    #Numeric preprocessing
    webtraffic = pd.read_table("web_traffic.tsv", header=None)
    numeric_preprocessing(webtraffic)

    #Image Hist Equalization
    img_hist = importImage("babon.jpg","babon image")
    img_hist_eq(img_hist)

    #IMG BG SEGMENTATION
    img_segmentation = importImage("apel.jpg","apel image")
    img_bg_segmentation(img_segmentation)








