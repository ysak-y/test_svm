from sklearn import datasets
from sklearn import svm
import numpy as np
from matplotlib import pyplot as plt

iris = datasets.load_iris()
digits = datasets.load_digits()

train_data = digits.data[:1500]
test_data = digits.data[1501:1796]
test_images = digits.images[1501:1796]

train_label = digits.target[:1500]
test_label = digits.target[1501:1796]

clf = svm.SVC()
clf.fit(train_data, train_label)
print(clf.predict(test_data))
pred = clf.predict(test_data)
bool_arr = (test_label == pred)

true_test_image = test_images[np.array(bool_arr == True)] 
true_test_label = test_label[np.array(bool_arr == True)] 

false_test_image = test_images[np.array(bool_arr == False)] 
false_test_label = pred[np.array(bool_arr == False)]

fig, ax = plt.subplots(20, 10, figsize=(20, 10), subplot_kw={'xticks': [], 'yticks': []})
fig.subplots_adjust(hspace=1, wspace=0.05)

for i in range(1, 20):
    for j in range(1, 10):
        if (i * j) - 1 >= len(true_test_image):
            break
        img = true_test_image[i * j - 1]
        ax[i - 1, j - 1].set_title(true_test_label[(i * j) - 1])
        ax[i - 1, j - 1].imshow(img, cmap=plt.get_cmap('gray'))

fig2, ax2 = plt.subplots(20, 10, figsize=(20, 10), subplot_kw={'xticks': [], 'yticks': []})
fig2.subplots_adjust(hspace=1.0, wspace=0.05)

for i in range(1, 20):
    for j in range(1, 10):
        if (i * j) - 1 >= len(false_test_image):
            break
        img = false_test_image[i * j - 1]
        ax2[i - 1, j - 1].set_title(false_test_label[(i * j) - 1])
        ax2[i - 1, j - 1].imshow(img, cmap=plt.get_cmap('gray'))

plt.show()
