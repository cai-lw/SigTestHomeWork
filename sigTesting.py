import numpy as np
from numpy import genfromtxt
from scipy.stats import ttest_ind, ttest_rel, ks_2samp, wilcoxon
import matplotlib.pyplot as plt
import csv


class SignificanceTesting(object):
    def __init__(self, filePath):
        self.filePath = filePath
        self.loadData()

    def loadData(self):
        models_scores = ['Baseline_R2', 'Baseline+Fusion_R2', 'Baseline+Ordering_R2',
                         'Baseline+Ordering+Fusion_R2', 'Baseline_RSU4', 'Baseline+Fusion_RSU4',
                         'Baseline+Ordering_RSU4', 'Baseline+Ordering+Fusion_RSU4']
        table = genfromtxt(self.filePath, delimiter=',', skip_header=True).transpose()
        self.data = dict(zip(models_scores, table))

    def ksTest(self, listA, listB):
        value, pvalue = ks_2samp(listA, listB)
        return pvalue

    def tTest(self, listA, listB):
        value, pvalue = ttest_ind(listA, listB)
        return pvalue

    def wilcoxonTest(self, listA, listB):
        T, pvalue = wilcoxon(listA, listB)
        return pvalue

    def writeOutput(self):
        w = 6
        h = 13
        resultsData = [[0 for x in range(w)] for y in range(h)] 
        resultsData[0] = ['metric','model', 'mean diff', 'P(T test)', 'P(wilcoxon test)' ,'P(ks test)']
        for row in range(1,7):
            resultsData[row][0]= 'ROUGE-2'
        for row in range(7,13):
            resultsData[row][0]= 'ROUGE-SU4'
        resultsData[1][1] = 'Baseline & Fusion'
        resultsData[2][1] = 'Baseline & Ordering'
        resultsData[3][1] = 'Fusion & Ordering'
        resultsData[4][1] = 'Baseline & Ordering+Fusion'
        resultsData[5][1] = 'Ordering & Ordering+Fusion'
        resultsData[6][1] = 'Fusion & Ordering+Fusion'
        resultsData[7][1] = 'Baseline & Fusion'
        resultsData[8][1] = 'Baseline & Ordering'
        resultsData[9][1] = 'Fusion & Ordering'
        resultsData[10][1] = 'Baseline & Ordering+Fusion'
        resultsData[11][1] = 'Ordering & Ordering+Fusion'
        resultsData[12][1] = 'Fusion & Ordering+Fusion'
        for row in range(1,13):
            modelA, modelB = resultsData[row][1].split('&')
            if row < 7:
                scoreType = 'R2'
            else:
                scoreType = 'RSU4'
            scoreA = modelA.strip() + '_' + scoreType
            scoreB = modelB.strip() + '_' + scoreType
            if not scoreA.startswith('Baseline'):
                scoreA = 'Baseline+' + scoreA
            if not scoreB.startswith('Baseline'):
                scoreB = 'Baseline+' + scoreB
            listA = self.data[scoreA]
            listB = self.data[scoreB]
            resultsData[row][2] = np.mean(listB - listA)
            resultsData[row][3] = self.tTest(listA, listB)
            resultsData[row][4] = self.wilcoxonTest(listA, listB)
            resultsData[row][5] = self.ksTest(listA, listB)

        with open('SigTestResultsInd.csv', 'w') as resultsFile:
            writer = csv.writer(resultsFile)
            writer.writerows(resultsData)

    def boxingPlot(self):
        plt.figure()
        plt.boxplot(np.array(list(self.data.values())).transpose(), labels=list(self.data.keys()))
        plt.xticks(fontsize=7, rotation=-20, ha='left')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    filePath = "ROUGE_SCORES.csv"
    sigInstance = SignificanceTesting(filePath)
    sigInstance.writeOutput()
    sigInstance.boxingPlot()

