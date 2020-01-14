import tensorflow as tf

# todo 结构化数据

# 特征列的选择依据：
#   1：特征类型
#       连续特征（生成DenseTensor）   -----  numeric_column
#                                    -----  indicator_column,embedding_column 基本就是做转换用的
#       分类(稀疏)特征（生成SparseTensor）  -----  categorical_column_with_*
#                                           categorical_column_with_vocabulary_list
#                                           categorical_column_with_vocabulary_file
#                                           ....
#                                           weighted_categorical_column
#                                           crossed_column(只能用于稀疏特征)
#       bucketized_column即可以用于 连续特征 ，也可以用于分类(稀疏)特征

#   2:模型类型
#       深度模型 (如：DNNClassifier，DNNRegressor..)
#            连续特征可以直接使用
#            稀疏特征不能直接使用，需要使用indicator_column或者embedding_column进行转换后使用
#       宽度（线性）模型(如：LinearClassifier，LinearRegressor..)
#            稀疏特征可以直接使用
#            连续特征如果使用线性模型官方推荐使用bucketized_column先进行分桶

user = {
    "name": ["kobe", "jame", "wade"],
    "no": [24, 23, 3],
    "son": [0, 2, 1]
}

# 数值列
nc_no = tf.feature_column.numeric_column(key="no")

# 分桶列
bc_no = tf.feature_column.bucketized_column(source_column=nc_no, boundaries=[10, 20])

# 分类列
cc_name = tf.feature_column.categorical_column_with_vocabulary_list(key="name",
                                                                    vocabulary_list=["kobe", "jame", "wade"])

# 转换 稀疏特征-->连续特征 （嵌入列，指示列）
ec_name = tf.feature_column.embedding_column(categorical_column=cc_name, dimension=2)
ic_name = tf.feature_column.indicator_column(categorical_column=cc_name)

# 组合列
cc_no_x_name = tf.feature_column.crossed_column(keys=[bc_no, cc_name], hash_bucket_size=10)

deep_feature_columns = [nc_no, ec_name]
wide_feature_columns = [bc_no, cc_name, cc_no_x_name]
# deep_model = tf.estimator.DNNRegressor(hidden_units=[10, 1], feature_columns=deep_feature_columns)
# wide_model = tf.estimator.LinearRegressor(feature_columns=wide_feature_columns)


make_parse_example_spec = tf.feature_column.make_parse_example_spec(deep_feature_columns)
print(make_parse_example_spec)
df = tf.keras.layers.DenseFeatures(feature_columns=deep_feature_columns)
print(df(user))
