import matplotlib
# matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
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
pre = []
re = []
pr_auc = []
for v in score:
    precision, recall, thresholds = precision_recall_curve(value, v)
    area = auc(recall, precision)
    pre.append(precision)
    re.append(recall)
    pr_auc.append(area)

test_class = ['Resnik', 'Zhang', 'BOG', 'SemFunSim', 'DNN-based regression model']#, 'DNN-based regression model'
colors = ['aqua', 'darkorange', 'cornflowerblue', 'darkslategray', 'crimson', 'darkgreen', 'darkblue', 'lavender']

plt.figure()
lw = 2

for i in range(len(test_class)):
    plt.plot(re[i], pre[i], color=colors[i],
             lw=lw, label='%s(area = %0.2f' % (test_class[i], pr_auc[i]*100)+r'%)')  ###假正率为横坐标，真正率为纵坐标做曲线

plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.title('Precision/Recall Curve')  # give plot a title
plt.xlabel('Recall')  # make axis labels
plt.ylabel('Precision')
plt.legend(loc="lower right")
plt.show()

# plt.savefig('p-r.png')