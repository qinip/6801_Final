from utils_zj import *
from matplotlib.pyplot import *
import numpy as np
import scipy.stats
import statsmodels.api as sm
from scipy import stats
import warnings

# def getBiasForOutlet(models, outlet, intervalName, dataSetIndex, dataDict,pearson=False):
#     """
#     given several embedding models data structure: {'2000-2004': {'nyt': <gensim.models.word2vec.Word2Vec at 0x1eb599c2550>,
#     and an outlet: nyt
#     an interval name: '2015-2019'
#     a dataSetIndex: 1-1
#     and a dataDict {'0-1': {'name': 'bipolar  Occupations',
#                   'constructPole1': ['man', 'men', 'male', 'males'],
#                   'constructPole2': ['woman', 'women', 'female', 'females'],
#                   'RealDataLexicons': [{'abandon': -1,

#     return the correlation bias and P value
#     """

#     try:  # Get the model for interval period and Specific outlet
#         model = models[intervalName][outlet]
#     except KeyError:
#         #if data doesn't exist return none
#         return None, None
#     constructPole1 = dataDict[dataSetIndex]['constructPole1']
#     constructPole2 = dataDict[dataSetIndex]['constructPole2']

#     constructPole1 = list(map(str.lower, constructPole1))
#     constructPole2 = list(map(str.lower, constructPole2))
#     constructPole1, constructPole2 = constructsFilter(model, constructPole1, constructPole2)
#                                                       #printFlag=False)  # Filter out terms not in model vocabulary

#     for RealDataTemp in dataDict[dataSetIndex]['RealDataLexicons']:
#         #         for RealDataTemp in [voteByEthnicity]:
#         RealData = realDataFilter(model, RealDataTemp,
#                                   0)  # Filter out terms not in model vocabulary (Last parameter Flag prints out OOV words)
#         Axis = dimensionN(model, constructPole1, constructPole2)
#         AxisName = 'Axis'
#         try:
#             dataFrame = makeDF(model, RealData.keys(), Axis, AxisName)
#         except:
#             continue
#         dataFrame['RealValues'] = RealData.values()
#         spearmanCorr, pearsonCorr = calculateCorrelations(dataFrame)# , printFlag=False)
#         if pearson==True:
#             try:
#                 return (pearsonCorr[0], pearsonCorr[1])  # Correlation and P value
#             except:
#                 print('outlet',pearsonCorr)
#                 return pearsonCorr,pearsonCorr
#         else:
#             try:
#                 return (spearmanCorr[0], spearmanCorr[1])  # Correlation and P value
#             except:
#                 print(outlet,spearmanCorr)
#                 return spearmanCorr,spearmanCorr

def getBiasForOutlet(models, outlet, intervalName, dataSetIndex, dataDict, pearson=False):
    """
    ...
    """
    try:
        model = models[intervalName][outlet]
    except KeyError:
        return None, None

    constructPole1 = dataDict[dataSetIndex]['constructPole1']
    constructPole2 = dataDict[dataSetIndex]['constructPole2']

    constructPole1 = list(map(str.lower, constructPole1))
    constructPole2 = list(map(str.lower, constructPole2))
    constructPole1, constructPole2 = constructsFilter(model, constructPole1, constructPole2)

    for RealDataTemp in dataDict[dataSetIndex]['RealDataLexicons']:
        RealData = realDataFilter(model, RealDataTemp, 0)
        Axis = dimensionN(model, constructPole1, constructPole2)
        AxisName = 'Axis'

        # Check if RealData is a dictionary and has at least 2 keys
        if not isinstance(RealData, dict) or len(RealData.keys()) < 2:
            continue

        # Check if Axis has at least 2 elements
        if len(Axis) < 2:
            continue

        try:
            dataFrame = makeDF(model, RealData.keys(), Axis, AxisName)
        except:
            continue

        dataFrame['RealValues'] = RealData.values()
        spearmanCorr, pearsonCorr = calculateCorrelations(dataFrame)

        if pearson:
            try:
                return pearsonCorr[0], pearsonCorr[1]
            except:
                print('outlet', pearsonCorr)
                return pearsonCorr, pearsonCorr
        else:
            try:
                return spearmanCorr[0], spearmanCorr[1]
            except:
                print(outlet, spearmanCorr)
                return spearmanCorr, spearmanCorr

    # If no valid dataFrame is found, return None
    return None, None

def getIntervalNames(startYear,endYear,intervalLength):
    """
    :param startYear: 2000
    :param endYear: 2020
    :param intervalLength: 5
    :return: ['2000-2004', '2005-2009', '2010-2014', '2015-2019']
    """

    intervalNames = [str(start)+'-'+str(start+intervalLength-1) for start in list(range(startYear,endYear,intervalLength))]
    return intervalNames


def getTimeSeriesOfBiasForOutlet(models,outlet,dataSetIndex,startYear,endYear,intervalLength,dataDict):
    """
    :param models: {'2000-2004': {'nyt': <gensim.models.word2vec.Word2Vec at 0x1eb599c2550>,
    :param outlet: 'nyt'
    :param dataSetIndex: '8-1'
    :param startYear: 2000
    :param endYear: 2019
    :param intervalLength: 5 --> 2000-2004, 2005-2009, 2010-2014...
    :param dataDict: {'0-1': {'name': 'bipolar  Occupations',
                  'constructPole1': ['man', 'men', 'male', 'males'],
                  'constructPole2': ['woman', 'women', 'female', 'females'],
                  'RealDataLexicons': [{'abandon': -1,
    :return: list (time series) of biases
    """
    intervalNames=getIntervalNames(startYear, endYear, intervalLength)

    biasesTimeSeries = []
    for intervalName in intervalNames:
    #     print(intervalName)
        c,p=getBiasForOutlet(models,outlet,intervalName,dataSetIndex,dataDict)
        biasesTimeSeries.append(c)
    return biasesTimeSeries


def getTimeSeriesOfBiasForSeveralOutlets(models,outlets,dataSetIndex,startYear,endYear,intervalLength,dataDict):
    intervalNames=getIntervalNames(startYear, endYear, intervalLength)
    m=np.zeros((len(outlets),len(intervalNames)))
    for outletIndex, outlet in enumerate(outlets):
        ts=getTimeSeriesOfBiasForOutlet(models,outlet,dataSetIndex,startYear,endYear,intervalLength,dataDict)
        m[outletIndex]=ts
    return m

def getAverageTimeSeriesOfBiasForSeveralOutlets(models,outlets,dataSetIndex,startYear,endYear,intervalLength,dataDict,z=1.96):
    intervalNames = [str(start)+'-'+str(start+intervalLength-1) for start in list(range(startYear,endYear,intervalLength))]
    m=np.zeros((len(outlets),len(intervalNames)))
    m[:] = np.nan
    for outletIndex, outlet in enumerate(outlets):
#         ts=getTimeSeriesOfBiasForOutlet(models,outlet,dataSetIndex,dataDict)
        ts=getTimeSeriesOfBiasForOutlet(models,outlet,dataSetIndex,startYear,endYear,intervalLength,dataDict)
        m[outletIndex] = ts
    mAverage = np.nanmean(m,axis=0) #Proper way when nan values are present
#     mAverage = np.nansum(m,axis=0)/m.shape[0] #
    mStd = np.nanstd(m,axis=0)
#     mStd = np.nanstd(m,axis=0)
    #Estimating confidence interval
#     n = len(outlets) # number of obs
    n = np.sum(~np.isnan(m),axis=0) #number of obs in each time interval
    z = 1.96 # for a 95% C
#     z = 2.58 # for a 99% C
#     z = 3.291 # for a 99.9% C
    CI = z * (mStd/np.sqrt(n))
    lowerCI = mAverage - CI
    upperCI = mAverage + CI
    return mAverage, CI



def getMultipleTimeSeriesOfBiasForOutlet(models,outlet,dataSetIndexes,startYear,endYear,intervalLength,dataDict):
    #Calculate Multiple time series For single outlet
    intervalNames = [str(start) + '-' + str(start + intervalLength - 1) for start in
                     list(range(startYear, endYear, intervalLength))]
    tsMatrixSingleOutlet = np.zeros((len(dataSetIndexes),len(intervalNames)))
    for dataSetIndexIndex,dataSetIndex in enumerate(dataSetIndexes):
        print(dataSetIndex,end=',')
        ts=getTimeSeriesOfBiasForOutlet(models,outlet,dataSetIndex,startYear,endYear,intervalLength,dataDict)
        tsMatrixSingleOutlet[dataSetIndexIndex]=ts
    return tsMatrixSingleOutlet

#Calculate Multiple time series Aggregating across outlets
def getMultipleTimeSeriesAggregatingAcrossOutlets(models,outlets,dataSetIndexes,startYear,endYear,intervalLength,dataDict):
    intervalNames = [str(start)+'-'+str(start+intervalLength-1) for start in list(range(startYear,endYear,intervalLength))]
    tsMatrix = np.zeros((len(dataSetIndexes),len(intervalNames)))
    CIMatrix = np.zeros((len(dataSetIndexes),len(intervalNames)))
    for dataSetIndexIndex,dataSetIndex in enumerate(dataSetIndexes):
        print(dataSetIndex,end=',')
        #Average of all outlets
        ts,CI=getAverageTimeSeriesOfBiasForSeveralOutlets(models,outlets,dataSetIndex,startYear,endYear,intervalLength,dataDict)

        tsMatrix[dataSetIndexIndex]=ts
        CIMatrix[dataSetIndexIndex]=CI
    return tsMatrix, CIMatrix

def testPvalue(yearsToPlot, correlationsTimeSeries):
    x = sm.add_constant(yearsToPlot)  ## model with intercept (beta_0)
    y = correlationsTimeSeries

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.OLS(y, x,missing='drop').fit()
        p_values = model.summary2().tables[1]['P>|t|']
        b1 = x1 = model.summary2().tables[1]['Coef.']['x1']
    pValueSlope = p_values[1]
    return b1, pValueSlope


def plotTimeSeriesForSingleOutlet(intervalNames,ts,CI=None,title=None):
    plt.plot(intervalNames,ts)
    if type(CI) == np.ndarray:
        fill_between(intervalNames, ts+CI, ts-CI,color='b', alpha=.2)
    plt.xticks(rotation=45)
    plt.title(title)


def plotTimeSeriesForSeveralOutlets(intervalNames, tsM, outlets, CIs=None, colors=['r', 'b'], title=None):
    for tsIndex, ts in enumerate(tsM):
        plt.plot(intervalNames, ts, label=outlets[tsIndex], color=colors[tsIndex])
        try:
            if CIs.any():
                fill_between(intervalNames, ts + CIs[tsIndex], ts - CIs[tsIndex], color=colors[tsIndex], alpha=.2)
        except:
            pass
    plt.xticks(rotation=45)
    plt.legend()
    plt.title(title)