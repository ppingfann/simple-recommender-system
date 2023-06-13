# simple-recommender-system
一个简单的推荐系统

# 基础架构图
![Image text](resources/picture/Infrastructure_diagram.png)

# 特征工程
## 原始数据源
原始数据集的来源地址之一：https://grouplens.org/datasets/
### Book-Crossing Dataset原始数据源
目录：rawdatafrombx
地址：http://www2.informatik.uni-freiburg.de/~cziegler/BX/
数据源文件：BX-Users.csv、BX-Books.csv、BX-Book-Ratings.csv
### MovieLens原始数据源
目录：rawdatafrommovielens
地址：https://movielens.org/
数据源文件：ratings.csv
### 原始数据源的整合
目录：raw_data
1. 将BX-Users源数据中的字段修改名称后写入到users.csv文件中。
2. 将BX-Books数据源中的字段修改名称并增加了一个自增ID后，写入到了books.csv文件中。
3. 因为BX-Book-Ratings数据源中缺少了用户的场景信息，因此，将ratings中的时间戳信息合并到了该数据源中，并生成了book_ratings.csv文件，
作为用户评分的原始数据源。
最终，得到了我们的原始数据源，共三个文件：
users.csv：用户信息。
books.csv：Item信息。
book_ratings.csv：用户对Item的评分。
## 训练以及测试数据集
目录：dataset
training_data_set.csv：训练数据集
validation_data_set.csv：验证数据集（暂时没有使用，因为还没有加入超参数的调整）
test_data_set.csv：测试数据集

后续也可以转为tfrecord格式，以加速Tensorflow训练时的数据集读取。
# 环境准备
## Tensorflow

# embedding
src/embedding/embedding.py 中做了item2vec的embedding过程示例。
可以看到，先对原始数据集进行了初步处理，将评分低于6分的过滤掉，并将剩余的Item按照评分大小进行排序，得到了每个用户针对Item的序列数据。
将以上数据送入mllib的Word2Vec函数进行处理。即可获得指定维度的向量了。

# 模型构建
构建模型的五个步骤：特征选择、模型设计、模型实现、模型训练、模型评估。
## Embedding+DNN
### 特征选择
|  特征分类   | 特征名称  | 特征字段  | 特征类别  | 处理方式  |
|  ----  | ----  | ----  | ----  | ----  |
| 用户特征  | 用户ID | User-ID | 类别型 | onehot+embedding |
| 用户特征  | 用户位置 | Location | 类别型 | onehot+embedding |
| 用户特征  | Age | Location | 类别型 | onehot+embedding |
| Item特征  | 图书索引 | ISBN | 类别型 | onehot+embedding |
| Item特征  | 书名 | Book-Title | 类别型 | onehot+embedding |
| Item特征  | 作者 | Book-Author | 类别型 | onehot+embedding |
| Item特征  | 发布年份 | Year-Of-Publication | 数值型 | 直接输入DNN |
| Item特征  | 出版方 | Publisher | 类别型 | onehot+embedding |
| 场景特征  | 评价时间 | timestamp | 数值型 | 直接输入DNN |
### 模型设计
选择了一个三层的MLP结构，其中前两层是128维的全连接层，最后一层采用单个sigmoid神经元作为输出层。
### 模型实现
参看embedding_dnn.py文件代码实现。
### 模型训练
我们将离散型特征的onehot以及embedding都放在模型中来处理，而数值型特征的归一化以及分桶都放在特征服务中来处理。
原因在于，离散型特征的onehot以及embedding需要一个目标。因此放在模型中来处理。
而数值型特征由于范围可能会变化，因此最好是在特征服务中来处理，而非放在模型中来处理。
### 模型训练结果
Test Loss -1.0968728878302678e+20, Test Accuracy 0.0014199770521372557, Test ROC AUC 0.5, Test PR AUC 0.37431004643440247

# 关于Keras
Keras 是一个用 Python 编写的高级神经网络 API，它能够以 TensorFlow, CNTK, 或者 Theano 作为后端运行。Keras 的开发重点是支持快速的实验。能够以最小的时延把你的想法转换为实验结果，是做好研究的关键。

简单来说，就是Keras提供了一个很方便的API接口，让我们可以高效的使用Tensorflow，且在2017年，Google的TensorFlow团队已在TensorFlow核心库中支持了Keras。

Keras官方网站：https://keras.io/
# 模型训练的结果
在训练完毕后，可以将模型保存起来，供在线推荐服务（如TFServing）加载使用。Python语句如：
```
tf.keras.models.save_model(
    model,
    model_save_path + "/20230216",
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)
```

保存到的目录中至少会有2个文件（目录）。其中：
`saved_model.pb`文件用于存储实际 TensorFlow 程序或模型，以及一组已命名的签名——每个签名标识一个接受张量输入和产生张量输出的函数。

`variables`目录下是一个或多个包含了模型权重的分片，格式一般为`variables.data-00000-of-00001`，还有一个用于存储哪些权重存储在哪些分片的索引文件，文件名称一般为`variables.index`。

用一句话概括的话，`saved_model.pb`存储模型，`variables`存储权重。

# 线上服务
## 开发语言选型
对于一个推荐系统线上服务来说，在进行开发语言选型时，主要考虑以下几个方面：
1. 用户请求响应时延低
2. 请求并发量高
3. 安全性
4. 生态系统
综合对比，决定使用Go语言进行开发。
## 开发框架选型
综合比较Golang开源的微服务框架，最终选择go-zero作为开发框架。
## 高并发
为了支撑线上的高并发请求，需要做的工作如下：
1. 负载均衡：在Kubernetes上进行服务的部署，并通过负载均衡器如CLB进行网关服务以及内部服务间的请求负载均衡。
2. 缓存：（1）当用户在相同上下文环境中多次请求时，可以利用缓存数据来减少服务器的计算压力。（2）所有的配置数据、特征数据、频控数据都用的是Redis进行存储，并且许多数据都缓存在了服务中，方便快速读取。
3. 熔断、限流、降级：在超出我们预设的流量峰值时，可以采用限流的方式随机抛弃一些请求，或采用降级的方式，给用户返回一些默认值。在极端情况下，也可以采用熔断机制，从而保护系统。
4. 服务间通信：服务对外采用的是http协议进行通信，而服务内部间的相互调用都使用的是RPC通信。

# 存储
## 存储工具选型的原则
在对存储工具进行选型时，有一个基本原则：把越频繁访问的数据放到读取越快的数据库甚至内存中，把越不常访问的数据放到便宜但查询速度较慢的数据库中。
## 特征数据的存储选型
我这里有三类特征数据，分别是用户特征、Item特征以及场景特征。
而且大多数的推荐系统都会有这三类特征数据。
场景特征：在线上服务进行特征访问时，场景特征是用户请求时带过来的，无需进行存储。
用户特征：用户特征一般量比较大（大公司都是十亿级的数据量），并且对于单个用户来说，访问推荐系统服务的频次不会特别高，但在读取该数据时，时延要求也是比较高的。因此用户特征可以存储在数据读取较快且不是特别昂贵的数据库中，比如：Memcached、Apache Cassandra、Redis等。
Item特征：有些场景的Item特征数量会比较大，而有些则比较小，这取决于我们的应用场景以及待推荐的物品情况。但一般来说，我们每次推荐都需要所有待推荐Item的特征，因此在Item数量较小的的场景下，我们可以将Item缓存入服务器内存中，以达到最小化读取特征时延的效果。