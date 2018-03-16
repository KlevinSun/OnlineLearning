import numpy as np
import math
import matplotlib.pyplot as plt

def staticexpert(dataMat, feature, b):      #dataMat is the cloud data except the 7th column, feature is the 7th column, and b is learning rate
    iterationNum, expertNum = np.shape(dataMat)
    weights = [1/expertNum for i in range(expertNum)]       #initialize experts's weight
    lossY = np.zeros(iterationNum)          #array for losses of each iterations
    totalLossY = 0
    lossresults = np.zeros(iterationNum)
    for i in range(iterationNum):           #iterations
        lossXY = np.zeros(expertNum)        #array for losses of each expert
        x = np.zeros(expertNum)
        weighttotal = 0.0
        for j in range(expertNum):          # calculation of each expert in ith iteration
            lossY[i] += weights[j] * dataMat[i, j]          # prediction
            lossXY[j] = (dataMat[i, j] - feature[i])**2     # loss of ith iteration jth expert
            x[j] = weights[j]*(math.e**((-b)*lossXY[j]))    # non-normalized weight distribution
            weighttotal += x[j]
        for k in range(expertNum):
            weights[k] = x[k]/weighttotal                   # normalized weight distribution
        totalLossY += lossY[i]
        lossresults[i] = totalLossY/(i+1)                   # loss of ith iteration
    return lossresults


def plot_aveloss(loss, b):                  # plot the average loss of learning
    dataArr = np.array(loss)
    m = np.shape(dataArr)[0]
    axis_x = []
    axis_y = []
    for i in range(m):
        axis_x.append(i)
        axis_y.append(dataArr[i])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(axis_x, axis_y, s=1, c='blue')
    plt.xlabel('iteration times learning rate b=' + repr(b)); plt.ylabel('current value of average loss');
    plt.savefig("loss when learning rate b=" + repr(b))
    plt.show()


def main():
    dataMat = np.loadtxt(open("C:\\test\\cloud.csv", "rb"), delimiter=",", skiprows=0)
    feature = dataMat.T[6]                      # the 7th column as the data feature
    print(feature)
    dataTrans = np.delete(dataMat.T, 6, axis=0)
    dataMat = dataTrans.T   # cloud data without 7th column
    learnrate = 1
    averageloss = staticexpert(dataMat, feature, learnrate)
    return averageloss, learnrate

if __name__ == "__main__":
    loss, b = main()
    print(loss.T)
    plot_aveloss(loss, b)
