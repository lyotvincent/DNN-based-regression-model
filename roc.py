# -*- coding: utf-8 -*-
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
import matplotlib.pyplot as plt
from itertools import cycle
value = []
score = []

# 获取真实值
with open('data\\test_data.csv') as f:
    line = f.readline()
    line = f.readline()
    while line:
        tmp = line.split(',')
        if len(score)==0:
            tmplist1 = []
            tmplist1.append(float(tmp[4]))
            score.append(tmplist1)
            tmplist2 = []
            tmplist2.append(float(tmp[5]))
            score.append(tmplist2)
            tmplist3 = []
            tmplist3.append(float(tmp[6]))
            score.append(tmplist3)
            tmplist4 = []
            tmplist4.append(float(tmp[7]))
            score.append(tmplist4)
            tmplist5 = []
            tmplist5.append(float(tmp[8]))
            score.append(tmplist5)
        else:
            score[0].append(float(tmp[4]))
            score[1].append(float(tmp[5]))
            score[2].append(float(tmp[6]))
            score[3].append(float(tmp[7]))
            score[4].append(float(tmp[8]))
        v = float(tmp[3])
        value.append(v)
        line = f.readline()

# 计算ROC曲线和AUC面积
fpr = []
tpr = []
roc_auc = []
for v in score:
    fprtmp, tprtmp, threshold = roc_curve(value, v)
    auctmp = auc(fprtmp, tprtmp)
    fpr.append(fprtmp)
    tpr.append(tprtmp)
    roc_auc.append(auctmp)

# 画图
#test_class = ['Resnik', 'Lin', 'Wang', 'Schlicher', 'BOG', 'FunSim', 'SemFunSim', 'me']
test_class = ['Resnik', 'Zhang', 'BOG', 'SemFunSim', 'DNN-based regression model']#, 'DNN-based regression model'
colors = ['aqua', 'darkorange', 'cornflowerblue', 'darkslategray', 'crimson', 'darkgreen', 'darkblue', 'lavender']

plt.figure()
lw = 2

for i in range(len(test_class)):
    plt.plot(fpr[i], tpr[i], color=colors[i],
             lw=lw, label='%s (area = %0.2f' % (test_class[i], roc_auc[i]*100)+r'%)')  ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Test Result')
plt.legend(loc="lower right")
plt.show()
