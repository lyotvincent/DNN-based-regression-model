import pandas as pd
from keras import *
from keras.layers import *
# import sys as sys
import datetime
import os
import MySQLdb
from sklearn.decomposition import PCA
import picture

vect_file = "data\\vectGene2.csv"
train_file = "data\\train_data.csv"
bench_file = "data\\bench_data.csv"


def r_square(y_true, y_pred):
    SSR = backend.mean(backend.square(y_pred - backend.mean(y_true)), axis=-1)
    SST = backend.mean(backend.square(y_true - backend.mean(y_true)), axis=-1)
    return SSR / SST


def get_vect_index(id, list):
    for i in range(len(list)):
        if list[i] == id:
            return i


def generate_vect_by_two_id(id1, id2, ids, vects):
    i = get_vect_index(id1, ids)
    j = get_vect_index(id2, ids)
    if i == None:
        return None
    if j == None:
        return None
    dis1 = vects.loc[[i, j]]
    dis1 = dis1.values.tolist()
    dis1 = dis1[0] + dis1[1]
    return dis1


def reinforce_train():
    # 产生训练数据: 基准集数据加上后新增数据
    # 获取基准集数据
    train_data = pd.read_csv(bench_file)

    train_data = train_data.sample(frac=1).reset_index(drop=True)
    # 区别特征X和目标Y
    x_data = []
    y_data = []

    # 读取疾病id以及对应向量
    vects = pd.read_csv(vect_file)
    # print(vects)
    ids = vects.pop('disease_id')

    for index, pair in train_data.iterrows():
        # print(pair)
        dis = generate_vect_by_two_id(pair['id1'], pair['id2'], ids, vects)
        if dis == None:
            continue
        x_data.append(dis)
        y_data.append(pair['y'])
        # 找到两疾病在DO中的公共节点的深度和最短距离,没办法遍历owl图，待定
        # depth, shortest_path = get_info_in_DO(pair)

    #降维pca
    di = 128
    pca = PCA(n_components=di)
    # pca = PCA(n_components=0.98)
    x_data = pca.fit_transform(x_data)
    # print(pca.n_components)

    # 获取测试数据
    test_file = "C:\\Users\\Administrator\\Desktop\\test_data.csv"
    # test_file = "C:\\Users\\Administrator\\Desktop\\UMNSRS_relatedness_DOID.csv"
    test_data = pd.read_csv(test_file)
    x_test = []
    y_test = []
    for index, pair in test_data.iterrows():
        test_dis = generate_vect_by_two_id(pair['id1'], pair['id2'], ids, vects)
        if test_dis == None:
            continue
        x_test.append(test_dis)
        y_test.append(pair['y'])
    # for i in range(2):
    #     print(x_test[i])
    x_test = pca.transform(x_test)

    # 增量学习
    # 判断文件不存在，即模型未训练
    if not os.path.exists(model_file):
        k = len(x_data[0])
        model = Sequential()
        model.add(Dense(36, activation='relu', input_shape=(k,), ))
        model.add(Dropout(0.5))
        model.add(Dense(18, activation='relu'))
        model.add((Dropout(0.5)))
        model.add(Dense(12, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        print('完成构建网络')
        # model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=[r_square])
        print('编译网络')
    else:
        model = models.load_model(model_file)
        print("记载" + updated_date + "训练模型。")

    # 训练网络
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    history = model.fit(x=x_data, y=y_data, batch_size=60, epochs=800, verbose=0,validation_data=(x_test, y_test))#, validation_split=0.2  ,validation_data=(x_test, y_test)
    print("模型训练完毕")


    # 第七步：查看模型评分
    # 查看模型评分
    loss = history.history['loss'][-1]
    r2 = history.history['r_square'][-1]
    val_loss = history.history['val_loss'][-1]
    val_r2 = history.history['val_r_square'][-1]
    print('loss=%.4f, R2=%.4f,val_loss=%.4f ,val_R2=%.4f' % (loss, r2, val_loss, val_r2))
    picture.figures(history)

    # 预测
    # 获取测试数据
    y_predict = model.predict(x_test)
    y_str = []
    for i in range(len(y_predict)):
        # y_predict[i] = str(y_predict[i])+'\n'
        strss = str(y_predict[i][0])
        y_str.append(strss+'\n')
    f1 = open('data\\test.txt', 'w')
    f1.writelines(y_str)
    f1.close()


reinforce_train()
