import tensorflow as tf

# Training samples path
TRAINING_SAMPLES_FILE_PATH = "../../resources/csv/dataset/training_data_set.csv"
# Test samples path
TEST_SAMPLES_FILE_PATH = "../../resources/csv/dataset/training_data_set.csv"
# Model save path
MODEL_SAVE_PATH = "../../resources/modelfile"


# load sample as tf dataset
def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=10,  # 指定了
        label_name='book_rating',  # 指定了CSV数据集中的标签列，即我们要预测的东西
        na_value="0",
        num_epochs=1,
        ignore_errors=True)
    return dataset


# split as test dataset and training dataset
train_dataset = get_dataset(TRAINING_SAMPLES_FILE_PATH)
test_dataset = get_dataset(TEST_SAMPLES_FILE_PATH)

# all categorical features
categorical_columns = []
user_col = tf.feature_column.categorical_column_with_identity(key='user_id', num_buckets=300000)
print("user_col\n")
print("user_col\n")
print(user_col)
user_emb_col = tf.feature_column.embedding_column(user_col, 10)
print("user_emb_col")
print(user_emb_col)
categorical_columns.append(user_emb_col)

# book id embedding feature
book_col = tf.feature_column.categorical_column_with_identity(key='book_id', num_buckets=100000)
book_emb_col = tf.feature_column.embedding_column(book_col, 10)
categorical_columns.append(book_emb_col)

# age embedding feature
age_col = tf.feature_column.categorical_column_with_identity(key='age', num_buckets=1000)
age_emb_col = tf.feature_column.embedding_column(age_col, 10)
categorical_columns.append(age_emb_col)
print("categorical_columns")
print(categorical_columns)
# all numerical features
# 被选择为label的字段，不能作为特征输入
numerical_columns = [tf.feature_column.numeric_column('year_of_publication'),
                     tf.feature_column.numeric_column('timestamp')]

# deepFM model architecture
model = tf.keras.Sequential([
    tf.keras.layers.DenseFeatures(numerical_columns + categorical_columns),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

# compile the model, set loss function, optimizer and evaluation metrics
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR')])

# train the model
model.fit(train_dataset, epochs=5)

# evaluate the model
test_loss, test_accuracy, test_roc_auc, test_pr_auc = model.evaluate(test_dataset)
print('\n\nTest Loss {}, Test Accuracy {}, Test ROC AUC {}, Test PR AUC {}'.format(test_loss, test_accuracy,
                                                                                   test_roc_auc, test_pr_auc))

# print some predict results
predictions = model.predict(test_dataset)
for prediction, goodRating in zip(predictions[:12], list(test_dataset)[0][1][:12]):
    print("Predicted good rating: {:.2%}".format(prediction[0]),
          " | Actual rating label: ",
          ("Good Rating" if bool(goodRating) else "Bad Rating"))

# save model
tf.keras.models.save_model(
    model,
    MODEL_SAVE_PATH + "/20230216",
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
) 
