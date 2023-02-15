# simple-recommender-system
一个简单的推荐系统

# 基础架构图
![Image text](resources/picture/Infrastructure_diagram.png)

# 原始数据源
http://www2.informatik.uni-freiburg.de/~cziegler/BX/
使用了其中三个数据源文件：BX-Users、BX-Books、BX-Book-Ratings


# 原始数据源的处理
1. 因为BX-Book-Ratings数据源中缺少了用户的场景信息，因此，将ratings中的时间戳信息合并到了该数据源中，并生成了book_ratings文件，
作为用户评分的原始数据源。

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
| 用户特征  | Age | Location | 数值型 | 直接输入DNN |
| Item特征  | 图书索引 | ISBN | 类别型 | onehot+embedding |
| Item特征  | 书名 | Book-Title | 类别型 | onehot+embedding |
| Item特征  | 作者 | Book-Author | 类别型 | onehot+embedding |
| Item特征  | 发布年份 | Year-Of-Publication | 数值型 | 直接输入DNN |
| Item特征  | 出版方 | Publisher | 类别型 | onehot+embedding |
### 模型设计