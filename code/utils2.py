import numpy as np
import pandas as pd 


def dataDictLoadLexicons(dataDict,Lexicon): #lexicon should be a list
	for k in dataDict.keys():
		dataDict[k]['RealDataLexicons'] = Lexicon
	return dataDict


def makeDFcosineSimilarity(model, word_list,constructPole1,constructPole2,dimensionName):
	#using average cosine similarity from word to pole construct instead of axis projections
	d=[]
	for word in word_list:
		cosineSimilaritiesWordToPole1 = []
		for wordInPole1 in constructPole1:
			cs = model.wv.similarity(word,wordInPole1) #cosine similarity
			cosineSimilaritiesWordToPole1.append(cs)
		cosineSimilaritiesWordToPole2 = []
		for wordInPole2 in constructPole2:
			cs = model.wv.similarity(word,wordInPole2) #cosine similarity
			cosineSimilaritiesWordToPole2.append(cs)
		meanCosineSimilarityToPole1 = np.array(cosineSimilaritiesWordToPole1).mean()
		meanCosineSimilarityToPole2 = np.array(cosineSimilaritiesWordToPole2).mean()
		d.append(meanCosineSimilarityToPole2-meanCosineSimilarityToPole1)
	df = pd.DataFrame({dimensionName: d}, index = word_list)
	return df

	