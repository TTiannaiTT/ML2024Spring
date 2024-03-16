import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import sklearn

def preProcess(y_score,threshold):
    y_pred = [0 for i in range(len(y_score))]
    # print(y_pred)
    for i in range(len(y_score)):
        if y_score[i] >= threshold:
            y_pred[i] = 1
    print(y_pred)
    return y_pred

# draw the confusion matrix
sns.set()
f,ax = plt.subplots()
y_true = [0,0,1,1,0,1,0,1,1,0]
y1_score = [0.38,0.28,0.67,0.38,0.11,0.43,0.88,0.54,0.29,0.75]
y2_score = [0.19,0.89,0.47,0.89,0.95,0.49,0.23,0.66,0.15,0.66]
print(y_true)

y1_pred = preProcess(y1_score,0.5)
y2_pred = preProcess(y2_score,0.5)
# 打印混淆矩阵
print("Confusion Matrix: ")

C2 = confusion_matrix(y_true,y1_pred,labels=[1,0])
#打印 C2
print(C2)
# sns.heatmap(C2,annot=True,xticklabels=[1,0], yticklabels=[1,0]) # draw
# ax.set_title('confusion matrix') #title
# ax.set_xlabel('predict') #x
# ax.set_ylabel('true') #y
# plt.show()
# plt.savefig("heatmap.png")
print(sklearn.metrics.f1_score(y_true, y2_pred, labels=None, pos_label=1, average='binary', sample_weight=None))

