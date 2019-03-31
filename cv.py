from sklearn.model_selection import StratifiedKFold,train_test_split
import pandas as pd
import keras
from keras import *
from keras.layers import *
from sklearn.decomposition import PCA

vect_file = "data\\vectGene2.csv"
train_file = "data\\train_data.csv"
bench_file = "data\\bench_data.csv"
model_file = r'D:\dis_ml\data_tmp\nn_model_last_train.h5'
chunk_size = 1000  # , chunksize=chunk_size

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


def r_square(y_true, y_pred):
    SSR = keras.backend.mean(keras.backend.square(y_pred - keras.backend.mean(y_true)), axis=-1)
    SST = keras.backend.mean(keras.backend.square(y_true - keras.backend.mean(y_true)), axis=-1)
    return SSR / SST


# 产生训练数据
# 获取基准集数据
train_data = pd.read_csv(bench_file, engine='python')
# 区别特征X和目标Y
x_data = []
y_data = []

# 读取疾病id以及对应向量
vects = pd.read_csv(vect_file, engine='python')
# print(vects)
ids = vects.pop('disease_id')

for index, pair in train_data.iterrows():
    # print(pair)
    dis = generate_vect_by_two_id(pair['id1'], pair['id2'], ids, vects)
    if dis == None:
        continue
    x_data.append(dis)
    y_data.append(pair['y'])

#降维pca
di = 128
pca = PCA(n_components=di)
# pca = PCA(n_components=0.99)
x_data = pca.fit_transform(x_data)
print(pca.n_components_)


# cross validation
# n-fold=5
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=47)
cvscores = []  # 交叉验证结果
for cnt,(train,test) in enumerate(skf.split(x_data,y_data)):
    k = len(x_data[0])
    model = Sequential()
    model.add(Dense(48, activation='relu', input_shape=(k,), ))
    model.add(Dropout(0.5))
    model.add(Dense(12, activation='relu'))
    model.add((Dropout(0.5)))
    # model.add(Dense(12, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    # model.summary()
    print('完成构建网络')
    # model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])
    model.compile(loss='mse', optimizer='adam', metrics=[r_square])
    print('编译网络')
    #注意,如何取数据!当然若是df型,df.iloc[train]取值
    x_train = x_data[train]
    x_test = x_data[test]
    y_train = np.array(y_data)[train]
    y_test = np.array(y_data)[test]
    # y_train = y_data[train]
    # y_test = y_data[test]
    history = model.fit(x_train, y_train,epochs=800,batch_size=60,verbose=0,)

    # scores的第一维是loss，第二维是acc
    scores = model.evaluate(x_test, y_test)
    print('[INFO] %s: %.4f' % (model.metrics_names[1], scores[1]))
    cvscores.append(scores[1])
cvscores = np.asarray(cvscores)
print('[INFO] %.4f (+/- %.4f)' % (np.mean(cvscores), np.std(cvscores)))