# ssl报错请运行以下命令，我是3.7 所以这里写3.7
# open /Applications/Python\ 3.7/Install\ Certificates.command
# https://timonweb.com/tutorials/fixing-certificate_verify_failed-error-when-trying-requests_html-out-on-mac/


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

proxy = 'http://127.0.0.1:1080'

os.environ['http_proxy'] = proxy
os.environ['HTTP_PROXY'] = proxy
os.environ['https_proxy'] = proxy
os.environ['HTTPS_PROXY'] = proxy

print(tf.__version__)
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 训练数据有60000 个 28 * 28 的三维结构, 训练标签同样也是有60000个
print(train_images.shape, len(train_labels))

# 训练标签的值为 0～9 的数字，对应了class_names 的十个分类
print(train_labels)

# 测试数据有10000 个 28 * 28 的三维结构, 测试标签也同样有10000个
print(test_images.shape, len(test_labels))

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# 在馈送到神经网络模型之前，我们将这些值缩放到0到1的范围。为此，我们将像素值值除以255。重要的是，对训练集和测试集要以相同的方式进行预处理:
train_images = train_images / 255.0
test_images = test_images / 255.0

# 显示训练集中的前25个图像，并在每个图像下方显示类名。验证数据格式是否正确，我们是否已准备好构建和训练网络。


plt.figure(figsize=(10, 10))

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# 网络中的第一层, tf.keras.layers.Flatten, 将图像格式从一个二维数组(包含着28x28个像素)转换成为一个包含着28 * 28 = 784个像素
# 的一维数组。可以将这个网络层视为它将图像中未堆叠的像素排列在一起。这个网络层没有需要学习的参数;它仅仅对数据进行格式化。
# 在像素被展平之后，网络由一个包含有两个tf.keras.layers.Dense网络层的序列组成。他们被称作稠密链接层或全连接层。
# 第一个Dense网络层包含有128个节点(或被称为神经元)。第二个(也是最后一个)网络层是一个包含10个节点的softmax层—它将返回包含10个概率分数
# 的数组，总和为1。每个节点包含一个分数，表示当前图像属于10个类别之一的概率。

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax),
])

# 编译模型
# 在模型准备好进行训练之前，它还需要一些配置。这些是在模型的编译(compile)步骤中添加的:

# 损失函数 —这可以衡量模型在培训过程中的准确程度。 我们希望将此函数最小化以"驱使"模型朝正确的方向拟合。
# 优化器 —这就是模型根据它看到的数据及其损失函数进行更新的方式。
# 评价方式 —用于监控训练和测试步骤。以下示例使用准确率(accuracy)，即正确分类的图像的分数。


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
# 训练神经网络模型需要以下步骤:
#
# 将训练数据提供给模型 - 在本案例中，他们是train_images和train_labels数组。
# 模型学习如何将图像与其标签关联
# 我们使用模型对测试集进行预测, 在本案例中为test_images数组。我们验证预测结果是否匹配test_labels数组中保存的标签。
# 通过调用model.fit方法来训练模型 — 模型对训练数据进行"拟合"。
model.fit(train_images, train_labels, epochs=5)

test_lost, test_acc = model.evaluate(test_images, test_labels)
# 事实证明，测试数据集的准确性略低于训练数据集的准确性。训练精度和测试精度之间的差距是过拟合的一个例子。
# 过拟合是指机器学习模型在新数据上的表现比在训练数据上表现更差。

print('Test accuracy:', test_acc)

predictions = model.predict(test_images)

# 预测是10个数字的数组。这些描述了模型的"信心"，即图像对应于10种不同服装中的每一种。我们可以看到哪个标签具有最高的置信度值：
# 因此，模型最有信心的是这个图像是ankle boot，或者 class_names[9]。 我们可以检查测试标签，看看这是否正确:


print(predictions[0], np.argmax(predictions[0]))


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.show()

i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.show()

num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, test_labels)
plt.show()

# 最后，使用训练的模型对单个图像进行预测。
img = test_images[0]
print(img.shape)

img = (np.expand_dims(img, 0))
print(img.shape)

predictions_single = model.predict(img)
print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.show()

prediction_result = np.argmax(predictions_single[0])
print(prediction_result)
