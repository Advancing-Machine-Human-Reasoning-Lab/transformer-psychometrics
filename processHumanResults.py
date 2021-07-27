# -*- coding: utf-8 -*-
"""
    Given the tsv of diagnostic results, computes various psychometrics scores.
    The simple command-line arguments have the following effects:
    onlyClustering: runs the k-medoids experiment.
    ablation: creates the per-model correlation heatmap
    doRanalysis: fits the Rasch models
    model: specifies which models to use 'all' includes all models
    
    If the args are left unchanged, the script will compute various psychometric properties of the items
    including simple problem difficulty.
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids

from statistics import mean
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, matthews_corrcoef, silhouette_score
from scipy.stats import pearsonr as pearsonrx, spearmanr as spearmanrx, pointbiserialr as pointbiserialrx, kendalltau, rankdata
from scipy.optimize import minimize
from sys import argv

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

"""
	implementations of k-medioids and driver code for getting results
"""
class ClusteringMethods():

	def __init__(self, doClusterProfiles = False):
		self.doClusterProfiles = doClusterProfiles

	@staticmethod
	def clusterQuestionsOptimized(scores, num_trials, usePearson=True):
		# use the implementation of k-medoids from sklearn-extra instead of custom one
		n = len(scores[0]) #number of items
		bestSC = 0
		bestClusters = None

		#scores is currently individuals, then individual scores. Convert
		#to items, and then individual scores on those items.
		items = [[] for _ in range(n)]
		for individual in scores:
			for i in range(n):
				items[i].append(individual[i])

		def itemDiff(i1, i2):
			"""
				Given the scores on two items, determines their distance
				this is the distance metric to be used
			"""
			if len(i1)!=len(i2):
				raise Exception("i1 and i2 are not evenly sized!")
			if usePearson:
				return 1 - Utilities.pearsonr(i1, i2)[0]
			else:
				return 1 - Utilities.spearmanr(i1, i2)[0]#math.sqrt(1 - spearmanr(i1, i2)[0])
		
		def GetClustering(items, labels, k):
			# kmedoids does not return actual clustered points, only assignments
			# so, convert to the format other scripts expect
			clusters = []
			for k_i in range(0,k):
				c = []
				for i in range(0,len(labels)):
					if labels[i] == k_i:
						c.append(i)
				clusters.append(c)
			
			return clusters

		# run the clustering x number of times
		# save the best observed clusters
		for i in range(0, num_trials):
			for k in range(2, n-1):
				try:
					kmedoids = KMedoids(n_clusters=k,metric=itemDiff,init='random',max_iter=300000).fit(items)
					labels = kmedoids.labels_
					sc = silhouette_score(items, labels,metric=itemDiff)
					if sc > bestSC: # we've found a new optimal k
						bestSC = sc
						bestClusters = GetClustering(items, labels, k)
				except ValueError:
					# for the human data some initalizations cannot be clustered
					# skip these and try again
					pass
				
		print("\t\tBest SC for k =", k, ":", bestSC)
			
		return [bestSC, bestClusters]
		



	
	@staticmethod
	def RunClustering(all_data, num_trials=500, doClusterProfiles=False, doClusterItems = True):
		
		if doClusterItems:
			for s in all_data:
				print("Item clustering for", s)
				allBestClusters = []
				for data in range(1,6):
					print("\t", ["H","LMT","LML","R",'LM'][data-1], ':')
					if data==5:
						# [bestScore, bestClusters] = ClusteringMethods.clusterQuestions(all_data[s][3]+all_data[s][4], usePearson=True)
						[bestScore, bestClusters] = ClusteringMethods.clusterQuestionsOptimized(all_data[s][3]+all_data[s][4], num_trials, usePearson=True)
					else:
						# [bestScore, bestClusters] = ClusteringMethods.clusterQuestions(all_data[s][data], usePearson=True)
						[bestScore, bestClusters] = ClusteringMethods.clusterQuestionsOptimized(all_data[s][data], num_trials, usePearson=True)
					for cluster in bestClusters:
						#print absolute item indices
						# pass
						print("\t\t", [all_data[s][0][c] for c in cluster])
					allBestClusters.append([[all_data[s][0][c] for c in cluster] for cluster in bestClusters if len(cluster)>0])
				#determine how well the best clusters align
				allPairs = list(combinations(all_data[s][0], 2))
				inSameCluster = []#[] for _ in allBestClusters]
				for bestClusters in allBestClusters:
					thisData = []
					for (i1,i2) in allPairs:
						loc1=-1
						loc2=-1
						for (l,C) in enumerate(bestClusters):
							if i1 in C:
								loc1=l
							if i2 in C:
								loc2=l
						if loc1==-1 or loc2==-1:
							print("Clusters:", bestClusters)
							print("Trying to find:", i1, i2)
							raise Exception("Couldn't find one!")
						if loc1==loc2:
							thisData.append(1)
						else:
							thisData.append(0)
					inSameCluster.append(thisData)
					# print(thisData)
				# for x in allBestClusters:
					# print('\n',x)
				print("\tBETWEEN H AND LMT:", matthews_corrcoef(inSameCluster[0], inSameCluster[1]), Utilities.pearsonr(inSameCluster[0], inSameCluster[1]))
				print("\tBETWEEN H AND LML:", matthews_corrcoef(inSameCluster[0], inSameCluster[2]), Utilities.pearsonr(inSameCluster[0], inSameCluster[2]))
				print("\tBETWEEN H AND LM:", matthews_corrcoef(inSameCluster[0], inSameCluster[4]), Utilities.pearsonr(inSameCluster[0], inSameCluster[4]))
				print("\tBETWEEN H AND R:", matthews_corrcoef(inSameCluster[0], inSameCluster[3]), Utilities.pearsonr(inSameCluster[0], inSameCluster[3]))

class Utilities():
	def __init__(self):
		pass

	@staticmethod
	def pearsonr(A,B):
		# return spearmanr(A,B)
		v = pearsonrx(A,B)
		if math.isnan(v[0]):
			return [0, v[1]]
		return v
	
	@staticmethod
	def spearmanr(A,B):
		v = spearmanrx(A,B)
		if math.isnan(v[0]):
			return [0, v[1]]
		return v
	
	@staticmethod
	def pointbiserialr(A,B):
		v = pointbiserialrx(A,B)
		if math.isnan(v[0]):
			return [0, v[1]]
		return v
	
	@staticmethod
	def MSE(L1,L2):
		"""Calculates mean squared error."""
		if len(L1)!=len(L2):
			raise Exception("Lists are not the same size!")
		return sum([(L1[i]-L2[i])*(L1[i]-L2[i]) for i in range(len(L1))])
	
	@staticmethod
	def likelihood(params, X, delta):
		"""
		calculates the log likelihood of the scores. Returns the *negative*
		log likelihood, since we'll be using the minimization function to calculate
		max log likelihood.
		"""
		theta = params
		total = 0
		for n in range(len(theta)): #for every individual
			for i in range(len(delta)): #for every item 
				t1 = X[n][i]*(theta[n]-delta[i])
				t2 = np.log(1 + np.exp(theta[n] - delta[i]))
				total += t1-t2
		return -1*total
	
	@staticmethod
	def getScores(T,L, indices):
		"""
		Used to quickly extract the scores of a specific set of questions (indices) from the transformer-
		and lstm-based models.
		"""
		for i in range(6, len(headers)):
			individualName = headers[i]
			myScores = [-1]*len(indices)
			for row in data:
				qid = int(row[0])
				if qid in indices:
					myScores[indices.index(qid)] = int(row[i])
			# print("For", individualName, "on morphological negation:")
			# for j in range(len(MN_indices)):
				# print('\t', MN_indices[j], ':', myScores[j])
			if -1 in myScores:
				raise Exception("Couldn't find one of the indices for an LM")
			if 'lstm' in individualName:
				L.append(myScores)
			else:#if 'roberta' in individualName:
				T.append(myScores)
	
	@staticmethod
	def getScoresofTransformer(T, L, model, indices):
		for i in range(6, len(headers)):
			individualName = headers[i]
			individualName = individualName.split("-")[0]
			myScores = [-1]*len(indices)
			for row in data:
				qid = int(row[0])
				if qid in indices:
					myScores[indices.index(qid)] = int(row[i])
			if -1 in myScores:
				raise Exception("Couldn't find one of the indices for an LM")
			elif "lstm" in individualName:
				L.append(myScores)
			elif model == individualName:
				T.append(myScores)

	@staticmethod
	def getScoresOfLM(lmName, indices):
		#gets the scores of a specific LM. Returns a list of scores (NOT a list of lists!).
		for i in range(6, len(headers)):
			individualName = headers[i].strip()
			if individualName != lmName:
				continue
			myScores = [-1]*len(indices)
			for row in data:
				qid = int(row[0])
				if qid in indices:
					myScores[indices.index(qid)] = int(row[i])
			if -1 in myScores:
				raise Exception("Couldn't find one of the indices for an LM")
			return myScores

class Psychometrics():
	def __init__(self):
		pass
	
	@staticmethod
	def makeAblationHeatMap():
		"""
			Given the results from the per-model transformer ablation, create a heatmap comparing
			each transformers performance aganist the mean correlation
		"""
		cats = ["morphological negation","prepositional phrases","lexical entailment","quantifiers","propositional structure", "richer logical structure", "world knowledge"]
		lmts = ["albert","bert","electra","longformer","roberta","spanbert","xlnet"]

		correlations = np.array([
			[-0.03,0.11,-0.23,-0.05,0.39,-0.08,-0.49],
			[0.06,-1.25,-0.12,0.12,-0.22,-0.45,0.09],
			[0,-0.77,-0.21,-0.11,-0.07,0.02,0.04],
			[0.18,-0.76,-0.56,0.01,-0.11,0.05,0.12],
			[-0.17,-1.19,-0.1,-0.11,-0.09,-0.25,-0.11],
			[0.25,0.53,0.01,0.34,-0.44,0.29,0.24],
			[-0.07,-0.95,-0.15,-0.12,-0.02,-0.26,-0.09]
		])

		
		fig, ax = plt.subplots()
		im = ax.imshow(correlations)

		# We want to show all ticks...
		ax.set_xticks(np.arange(len(lmts)))
		ax.set_yticks(np.arange(len(cats)))
		# ... and label them with the respective list entries
		ax.set_xticklabels(lmts)
		ax.set_yticklabels(cats)
		# matplotlib.colorbar.ColorbarBase(ax=ax,cmap=matplotlib.colors.Colormap(name="seismic"))
		

		# Rotate the tick labels and set their alignment.
		plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
				rotation_mode="anchor")

		# Loop over data dimensions and create text annotations.
		for i in range(len(cats)):
			for j in range(len(lmts)):
				text = ax.text(j, i, correlations[i, j],
							ha="center", va="center", color="red")


		fig.tight_layout()
		plt.show()

	@staticmethod
	def calcAllIIC(S, isBinary=True):
		"""Calculates all inter-item correlations. Calculates correlations between *every* pair of questions (using phi coefficient / MCC if isBinary==True; otherwise we use Spearman correlation, following the default choice here: https://rdrr.io/cran/performance/man/item_intercor.html), so that the average of them can be used (to calculate average IIC). "The ideal range of average inter-item correlation is 0.15 to 0.50; less than this, and the items are not well correlated and don’t measuring the same construct or idea very well (if at all). More than 0.50, and the items are so close as to be almost repetitive." Another source says they should be between 0.2 and 0.4.
		GIVEN: a list of scores [ [c11, c12, ..., c1n], [c21, c22, ..., c2n], ..., [cm1, cm2, ..., cmn]] where:
			Each cij is the score participant i got on question j correct.
			If isBinary==True, then all cij must be 0 or 1. 

		RETURNS: [a_1_2, ..., a_n-1_n] where a_i_j is the inter-item correlation between questions i and j. The size of this returned list is (n-1)+(n-2)+...+2+1 = n(n+1)/2.
		"""
		n = set([len(row) for row in S])
		if len(n)>1:
			raise Exception("The number of elements in each list in S must be the same!")
		n = list(n)[0] #n = number of questions
		m = len(S) #m = number of individuals

		toReturn = []
		for i in range(n):
			for j in range(i+1,n):
				scoresI = [P[i] for P in S]
				scoresJ = [P[j] for P in S]
				if isBinary:
					toReturn.append(matthews_corrcoef(scoresI, scoresJ))
				else:
					print("\n\n\n\nSCORESI:", scoresI)
					print("\n\nSCORESJ:", scoresJ)
					toReturn.append(Utilities.pearsonr(scoresI, scoresJ)[0])
		return toReturn

	@staticmethod
	def calcCronbachAlpha(scores, isBinary=True):
		"""Calculates Cronbach's alpha. 

		GIVEN: a list of scores [ [c11, c12, ..., c1n], [c21, c22, ..., c2n], ..., [cm1, cm2, ..., cmn]] where:
			Each cij is the score participant i got on question j correct.
			If isBinary==True, then all cij must be 0 or 1. Otherwise, they must be between 0 and 1.
		
		RETURNS: Cronbach's Alpha.
		"""
		n = set([len(row) for row in scores])
		if len(n)>1:
			raise Exception("The number of elements in each list in S must be the same!")
		k = len(scores[0])
		#calculate sum of component variances
		allVariances = []
		for i in range(len(scores[0])):
			allVariances.append( np.var([itemScores[i] for itemScores in scores]) )
		totalVar = np.var([sum(s) for s in scores])
		sumC = sum(allVariances)
		# input("totalVar:" + str(totalVar) + ", allVariances:" + str(allVariances))
		toReturn = (k/(k-1))*(1 - (sumC/totalVar))
		# if math.isnan(toReturn):
		# 	print("isnan! k:", k, "sumC:", sumC, "totalVar", totalVar)
		# 	# exit()
		# elif math.isinf(toReturn):
		# 	print("isinf! k:", k, "sumC:", sumC, "totalVar", totalVar)
		# 	# exit()
		return toReturn
	
	@staticmethod
	def outputDifficulty(name, indices, model):
		print('\t',name,':')
		H_Diff = [mean(allScoresFromQuestionId[i]) for i in indices]
		LMT_Diff,LML_Diff = [[],[]]
		if model == "all":
			Utilities.getScores(LMT_Diff,LML_Diff,indices)
		else:
			Utilities.getScoresofTransformer(LMT_Diff, LML_Diff, model, indices)
		# if keepTop30: #remove all but top 30 performers in LMT_Diff
		# 	while len(LMT_Diff)>30:
		# 		scores = [sum(I) for I in LMT_Diff]
		# 		i = scores.index(min(scores))
		# 		LMT_Diff.pop(i)		
		LM_Diff = LMT_Diff + LML_Diff
		#refactor because these are individuals then questions, rather than questions then individuals.
		LMT_Diff = [mean([individual[i] for individual in LMT_Diff]) for i in range(len(LMT_Diff[0]))]
		LML_Diff = [mean([individual[i] for individual in LML_Diff]) for i in range(len(LML_Diff[0]))]
		LM_Diff = [mean([individual[i] for individual in LM_Diff]) for i in range(len(LM_Diff[0]))]
		R_Diff = [mean([random.randint(0,1) for j in range(20)]) for i in range(len(H_Diff))]
		print('\t\tVariances:', 
			mean([np.var(allScoresFromQuestionId[i]) for i in indices]))
		print('\t\tFrom H to LMT:', Utilities.spearmanr(H_Diff, LMT_Diff), Utilities.MSE(H_Diff, LMT_Diff))
		print('\t\tFrom H to LML:', Utilities.spearmanr(H_Diff, LML_Diff), Utilities.MSE(H_Diff, LML_Diff))
		print('\t\tFrom H to LM:', Utilities.spearmanr(H_Diff, LM_Diff), Utilities.MSE(H_Diff, LM_Diff))
		print('\t\tFrom H to R:', Utilities.spearmanr(H_Diff, R_Diff), Utilities.MSE(H_Diff, R_Diff))

	@staticmethod
	def getAllDiscriminationIndexes(scores):
		#Returns a list with the discrimination index of each item
		allScores = [sum(S)/len(S) for S in scores]
		# print("all scores:", allScores)
		allScoresSorted = sorted(allScores)
		uniqueScores = set(allScoresSorted)
		isUpper = [False for s in scores] #True if it's in the upper group
		#assign all individuals into the higher group or the lower group, unless they're exactly in the middle.
		#following directions from here: https://fcit.usf.edu/assessment/selected/responsec.html
		if len(uniqueScores)==1:
			raise Exception("Cannot split scores into two groups!")
		if len(uniqueScores)==2:
			for i in range(isUpper):
				isUpper[i] = (allScores[i]==allScoresSorted[0])
		else:
			#start in the middle index (int((len-1)/2). Go in each direction and find when the value
			#first changes, so we know where to split the high and low groups.
			curr = int((len(allScores)-1)/2)
			fPass = curr
			bPass = curr
			currVal = allScoresSorted[fPass]
			#forward pass
			while fPass < len(allScores):
				if allScoresSorted[fPass] != currVal:
					break
				currVal = allScoresSorted[fPass]
				fPass += 1
			currVal = allScoresSorted[bPass]
			while bPass >= 0:
				if allScoresSorted[bPass] != currVal:
					break
				currVal = allScoresSorted[bPass]
				bPass -= 1
			#which will lead to a more even division, fPass or bPass?
			wasFPass = True
			if fPass-(len(allScores)/2) > (len(allScores)/2)-bPass:
				changingPoint = allScoresSorted[fPass]
			else:
				changingPoint = allScoresSorted[bPass]
				wasFPass = False
			#now assign the individuals to groups
			for i in range(len(allScores)):
				if wasFPass:
					isUpper[i] = (allScores[i] >= changingPoint)
				else:
					isUpper[i] = (allScores[i] > changingPoint)
		# print("is upper:", isUpper)
		# now that we've split them into groups, calculate item discrimination index.
		#Determine the Discrimination Index by subtracting the number of students in the lower group who got the item correct from the number of students in the upper group who got the item correct.  Then, divide by the number of students in each group (in this case, there are five in each group).
		toReturn = []
		numUpper = sum(isUpper)
		numLower = len(isUpper) - numUpper
		for i in range(len(scores[0])):
			u = 0
			l = 0
			for s in range(len(scores)):
				#did this person get item i correct?
				if scores[s][i]==1:
					if isUpper[s]:
						u+=1
					else:
						l+=1
			dIndex = (u-l)*2/len(scores)
			toReturn.append(dIndex)
		# for x in zip(allScores, isUpper, toReturn):
		# 	print(x)
		return toReturn

	@staticmethod
	def getAllDiscriminationIndexes27(scores):
		#another method which only counts top/bottom 27% as the groups to consider.
		#from: https://ppukmdotorg.wordpress.com/2015/04/02/calculating-omr-indexes/
		#Returns a list with the discrimination index of each item
		allScores = [sum(S)/len(S) for S in scores]
		# print("all scores:", allScores)
		allScoresSorted = sorted([(s,i) for (i,s) in enumerate(allScores)])
		bottomGroupIndices = [i for (s,i) in allScoresSorted[:int(len(allScoresSorted)*0.27)+1]]
		topGroupIndices = [i for (s,i) in allScoresSorted[len(allScoresSorted) - int(len(allScoresSorted)*0.27)+1:]]
		isUpper = [False for s in scores] #True if it's in the upper group	
		# now that we've split them into groups, calculate item discrimination index.
		#EXAMPLE: Since we have 22 students in this example, 27% out of 22 = 6 students. So we will have to take the number of correct answers from 6 of the top students (H) and deduct the number of correct answers from 6 of the bottom students (L), then divide it by 6.
		toReturn = []
		for i in range(len(scores[0])): #for every problem
			u = 0
			l = 0
			for s in bottomGroupIndices:
				l += scores[s][i]
			for s in topGroupIndices:
				u += scores[s][i]
			toReturn.append((u-l)/len(topGroupIndices))
		return toReturn


	#corrected item-total correlation
	@staticmethod
	def correctedITC(scores):
		"""Calculates item-total correlation. Returns a list [i1, i2, ..., in] where ix is
		the item-total correlation of the xth item."""
		n = set([len(row) for row in scores])
		if len(n)>1:
			raise Exception("The number of elements in each list in S must be the same!")
		n = len(scores[0]) #n = number of questions
		m = len(scores) #m = number of individuals

		toReturn = []
		for i in range(n):
			itemScores = [individual[i] for individual in scores] #everyone's score on item i
			totalScores = [sum(individual)-individual[i] for individual in scores]
			toReturn.append(Utilities.pointbiserialr(itemScores, totalScores)[0])
		return toReturn

	@staticmethod
	def reportITC(name, H_scores, LMT_scores, LML_scores, R_scores):
		print('\t', name, '(pearson corr. and MSE):')
		H_ITC = Psychometrics.correctedITC(H_scores)
		LMT_ITC = Psychometrics.correctedITC(LMT_scores)
		LML_ITC = Psychometrics.correctedITC(LML_scores)
		LM_ITC = Psychometrics.correctedITC(LMT_scores + LML_scores)
		R_ITC = Psychometrics.correctedITC(R_scores)
		# print('\t\tFrom H to H:', Utilities.pearsonr(H_ITC, H_ITC)[0], Utilities.spearmanr(H_ITC, H_ITC)[0])
		print('\t\tFrom H to LMT:', Utilities.pearsonr(H_ITC, LMT_ITC), Utilities.MSE(H_ITC, LMT_ITC))
		print('\t\tFrom H to LML:', Utilities.pearsonr(H_ITC, LML_ITC), Utilities.MSE(H_ITC, LML_ITC))
		print('\t\tFrom H to LM:', Utilities.pearsonr(H_ITC, LM_ITC), Utilities.MSE(H_ITC, LM_ITC))
		print('\t\tFrom H to R:', Utilities.pearsonr(H_ITC, R_ITC), Utilities.MSE(H_ITC, R_ITC))

		print("\t\tH_ITC:", H_ITC)
		print("\t\tLMT_ITC:", LMT_ITC)
		print("\t\tR_ITC:", R_ITC)

		# plt.plot(H_IIC, LMT_IIC, 'o', color='black')
		# plt.title(name + ": LMT (y) vs. H (x)")
		# plt.show()

	@staticmethod
	def reportIIC(name, H_scores, LMT_scores, LML_scores, R_scores):
		"""Calculates the correlation between every single pair of items, and then determines
		how similar those correlations are when using human vs. non/human scores."""
		print('\t', name, 'Pairwise IIC (pearson corr. and MSE):')
		H_IIC = Psychometrics.calcAllIIC(H_scores)
		LMT_IIC = Psychometrics.calcAllIIC(LMT_scores)
		LML_IIC = Psychometrics.calcAllIIC(LML_scores)
		LM_IIC = Psychometrics.calcAllIIC(LMT_scores + LML_scores)
		R_IIC = Psychometrics.calcAllIIC(R_scores)
		# print('\t\tFrom H to H:', Utilities.pearsonr(H_IIC, H_IIC)[0], Utilities.spearmanr(H_IIC, H_IIC)[0])
		print('\t\tFrom H to LMT:', Utilities.pearsonr(H_IIC, LMT_IIC), Utilities.MSE(H_IIC, LMT_IIC))
		print('\t\tFrom H to LML:', Utilities.pearsonr(H_IIC, LML_IIC), Utilities.MSE(H_IIC, LML_IIC))
		print('\t\tFrom H to LM:', Utilities.pearsonr(H_IIC, LM_IIC), Utilities.MSE(H_IIC, LM_IIC))
		print('\t\tFrom H to R:', Utilities.pearsonr(H_IIC, R_IIC), Utilities.MSE(H_IIC, R_IIC))
	
	"""
	SIMULATE ITEM REMOVAL:
		We take the 15 questions in a subcategory, and then use the human data to check 
		which item has the lowest item-total correlation. Reduce until we have 5 questions,
		calculate the Average IIC, Average ITC, and Cronbach's Alpha, and call these the
		optimal scores.
		Next, using LM (or R) data only, do the same reduction to five questions. Calculate
		Average IIC, Average ITC, Cronbach's, and compare to optimal scores.

		Then, we do a similar experiment, except instead of using a greedy method to remove
		items, we check every possible subset of 5 items and obtain the global maximum.
	"""

	@staticmethod
	def performItemRemoval_greedy(scores, Hscores):
		#create deep copies
		# indices2 = [x for x in indices]
		scores2 = [[s for s in I] for I in scores]
		Hscores2 = [[s for s in I] for I in Hscores]
		while len(scores2[0]) > 5:
			citcScores = Psychometrics.correctedITC(scores2)
			iToRemove = citcScores.index(min(citcScores))
			# indices2.pop(iToRemove)
			for individualScores in scores2:
				individualScores.pop(iToRemove)
			for individualScores in Hscores2:
				individualScores.pop(iToRemove)
		#compile human responses for the indices that remain
		return [mean(Psychometrics.calcAllIIC(Hscores2)), mean(Psychometrics.correctedITC(Hscores2)), Psychometrics.calcCronbachAlpha(Hscores2)]
	
	@staticmethod
	def performItemRemoval_global(scores, Hscores, metric=0):
		"""
		metric = which to use to select global optimum:
		0 - use AIIC
		1 - use AITC
		2 - use Cronbach's alpha
		"""
		numQuestions = len(scores[0])
		bestIndices = [0,1,2,3,4]
		bestScore = -10000
		for i1 in range(numQuestions):
			for i2 in range(i1+1, numQuestions):
				for i3 in range(i2+1, numQuestions):
					for i4 in range(i3+1, numQuestions):
						for i5 in range(i4+1, numQuestions):
							# indices2 = [indices[i] for i in [i1,i2,i3,i4,i5]]
							indices = [i1,i2,i3,i4,i5]
							scores2 = [[individual[i] for i in indices] for individual in scores]
							s = [mean(Psychometrics.calcAllIIC(scores2)), mean(Psychometrics.correctedITC(scores2)), Psychometrics.calcCronbachAlpha(scores2)][metric]
							if s>bestScore:
								bestScore = s
								bestIndices = [i for i in indices]
		#now what is the score using human data?
		# print('\t\t\t\tBest indices were', bestIndices)
		Hscores2 = [[individual[i] for i in bestIndices] for individual in Hscores]
		return [mean(Psychometrics.calcAllIIC(Hscores2)), mean(Psychometrics.correctedITC(Hscores2)), Psychometrics.calcCronbachAlpha(Hscores2)][metric]
	
	@staticmethod
	def calcIndividualPositions(scores):
		"""Calculates individual scores based on rasch modeling, and returns [I, Delta, Theta, Theta2] where:
		I = a list of the five item indices (in scores) that are determined to be best as anchor questions. 
		Delta = the item positions assigned to the items in I.
		Theta_full = the scores assigned to each individual based on the full set of item positions.
		Theta_reduced = the scores assigned to each individual based on the item positions of items in I.
		"""
		#calculate difficulty using simple difficulty
		delta_all = [mean([individual[i] for individual in scores]) for i in range(len(scores[0]))]
		#using max log likelihood, assign individual scores Theta
		theta_full = minimize(Utilities.likelihood, 
						[0.5 for x in scores], 
						args=(scores, delta_all), 
						method='L-BFGS-B', 
						bounds=[(-3,3)]*len(scores))['x']
		#now that we have the individual positions when the entire question set is considered, let's find
		#the five questions which best approximate this.
		theta_reduced = None
		indices_reduced = None
		bestFitScore = -10000
		for combo in combinations(range(len(scores[0])), 5):
			#only keep the scores for items indexed in combo
			subsetScores = [[individual[i] for i in combo] for individual in scores]
			theta_new = minimize(Utilities.likelihood, 
							[0.5 for x in subsetScores], 
							args=(subsetScores, [delta_all[i] for i in combo]), 
							method='L-BFGS-B', 
							bounds=[(-3,3)]*len(subsetScores))['x']
			#calculate how well these positions match those in theta_full
			thisScore = Utilities.pearsonr(theta_full, theta_new)[0]
			#convert theta_full and _reduced to lists of rankings
			# thisScore = kendalltau(rankdata(theta_full, method='min'),rankdata(theta_new,method='min'))[0]
			if thisScore > bestFitScore:
				bestFitScore = thisScore
				theta_reduced = theta_new
				indices_reduced = list(combo)
		return [indices_reduced, [delta_all[i] for i in indices_reduced], theta_full, theta_reduced]


if __name__ == "__main__":
	print('\n'*10)

	# parse experiment args, which should be passed in this exact order
	try:
		onlyKeepTop = bool(argv[1]) 			# Only keep the top 20 LMs 
		removePerfect = bool(argv[2])		    # Don't include human or LM results with 100% scores
		discrimination_index = bool(argv[3])    # Run calculations for discrimination index
		metric = int(argv[4])
		onlyClustering = bool(argv[5])			# Whether to run clustering experiments
		doRanalysis = bool(argv[6])
		num_cluster_trials = int(argv[7])
		model = str(argv[8])
		ablation = bool(argv[9])

	except Exception:
		print("ERROR: Not enough args or args were the wrong type.\nUsing defaults.")
		onlyKeepTop = False
		removePerfect = False
		discrimination_index = False
		metric = 2
		onlyClustering = False
		doRanalysis = False
		num_cluster_trials = 500
		model = "all"
		ablation = False
	
	#load scores from LMs
	#assume that the empty column is removed, and last line is summation stuff
	with open("Diagnostic Results.tsv",encoding='utf-8') as F:
		data = [[v.strip() for v in l.strip().split('\t')] for l in F.readlines()]
	headers = [v.strip() for v in data.pop(0)]
	buildsOn = [v.strip() for v in data.pop(0)]
	data.pop(len(data)-1) #remove summation row

	qsToKeep = set([184,742,194,185,743,195,188,752,196,189,753,197,190,198,191,76,74,499,75,77,78,79,80,81,82,83,84,85,
				498,506,347,350,346,348,351,420,349,422,421,352,423,424,353,522,425,130,220,131,221,230,222,231,223,234,
				224,235,225,237,226,239,532,540,530,533,541,531,969,836,538,837,539,772,773,834,835,565,688,564,566,689,
				580,567,692,581,590,693,671,591,704,674,390,410,394,391,411,402,392,722,412,393,723,487,395,758,606,273,
				572,6,274,573,7,275,272,277,276,278,280,279,284,281,293,296,292,295,297,294,299,920,298,300,921,302,301,
				303,304,53,68,52,58,69,61,59,70,494,60,71,495,505,504,809,446,447,790,740,838,741,839,786,950,787,951,788,
				789,791,810,204,202,200,205,203,201,315,320,314,316,321,318,317,324,322,10,22,11,23,12,24,13,25,14,26,15,
				27,16,28,17,110,32,111,33,114,108,115,109,116,112,117,113,122,120,123,435,618,434,497,619,496,612,638,614,
				613,639,640,615,846,746,2,0,354,3,1,358,355,4,367,359,5,371,362,8,429,46,36,44,47,37,45,88,38,86,89,39,87,96,40,90])
	


	data = [d for d in data if int(d[0]) in qsToKeep]
	# subcats = set([row[1] for row in data])b
	#put all scores into necessary formats
	#dictionary of lists, such that the key is the id of a question, and it retrieves a list of of the scores of all individual people (only including those who answered all parts of the survey). 
	#So for example, allScoresFromQuestionId[17] should return a list [s1, s2, ..., sn] where si is 1 iff participant i got question 17 correct. 
	# The order of the list should be consistent as well, so allScoresFromQuestionId[28] should return a list with the participants reported in the exact same order as before.
	allScoresFromQuestionId = {10: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], 
							22: [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], 
							11: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							23: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							12: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], 
							24: [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0], 
							13: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], 
							25: [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							14: [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], 
							26: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							15: [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], 
							27: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							16: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							28: [1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0], 
							17: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							130: [0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1], 
							220: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							131: [1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1], 
							221: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							230: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
							222: [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0], 
							231: [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
							223: [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							234: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
							224: [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							235: [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], 
							225: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], 
							237: [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1], 
							226: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							239: [1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1], 
							46: [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1], 
							36: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							44: [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							47: [1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1], 
							37: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							45: [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							88: [1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1], 
							38: [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], 
							86: [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1], 
							89: [1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1], 
							39: [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], 
							87: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1], 
							40: [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0], 
							90: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							68: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							52: [1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1], 
							58: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
							69: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							61: [1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1], 
							53: [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1], 
							59: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
							70: [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							494: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							60: [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1], 
							71: [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1], 
							495: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							505: [1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1], 
							504: [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0], 
							809: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
							2: [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
							0: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							354: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							3: [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
							1: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							358: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							355: [0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1], 
							4: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							367: [1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1], 
							359: [0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1], 
							5: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							371: [0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0], 
							362: [0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1], 
							8: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							429: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
							347: [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1], 
							350: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							346: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							348: [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1], 
							351: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							420: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							349: [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1], 
							422: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							421: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							352: [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1], 
							423: [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							424: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							353: [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1], 
							522: [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], 
							425: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
							565: [0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1], 
							688: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
							564: [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], 
							566: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1], 
							689: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
							580: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1], 
							567: [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1], 
							692: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
							581: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1], 
							590: [0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1], 
							693: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
							671: [0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1], 
							591: [0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1], 
							704: [1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0], 
							674: [1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1]}
	

	#morphological negation (phase 1)
	MN_indices = [10, 22, 11, 23, 12, 24, 13, 25, 14, 26, 15, 27, 16, 28, 17] 
	MN_H_scores = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], 
				[1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], 
				[1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], 
				[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], 
				[1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1], 
				[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1], 
				[1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1], 
				[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]]
	
	MN_LMT_scores = [] #LM (transformer-based) scores
	MN_LML_scores = [] #LM (LSTM-based) scores
	if model == "all":
		Utilities.getScores(MN_LMT_scores, MN_LML_scores, MN_indices)
	else:
		Utilities.getScoresofTransformer(MN_LMT_scores, MN_LML_scores, model, MN_indices)
	MN_R_scores = [[random.randint(0,1) for i in range(len(MN_H_scores[0]))] for j in range(20)] #random scores

	#prepositional phrases (phase 2)
	PP_indices = [130, 220, 131, 221, 230, 222, 231, 223, 234, 224, 235, 225, 237, 226, 239]
	PP_H_scores = [[1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1], 
				[1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1], 
				[0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0], 
				[0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0], 
				[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1], 
				[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], 
				[0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1], 
				[0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1], 
				[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], 
				[1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1], 
				[1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1], 
				[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1], 
				[1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1], 
				[0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1], 
				[0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0], 
				[1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0], 
				[0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0], 
				[1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1], 
				[0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1], 
				[1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], 
				[1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1], 
				[0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0], 
				[0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1], 
				[0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], 
				[0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1], 
				[0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0], 
				[1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1], 
				[1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], 
				[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1], 
				[0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], 
				[0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1], 
				[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], 
				[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1]]
	
	PP_LMT_scores = [] #LM (transformer-based) scores
	PP_LML_scores = [] #LM (LSTM-based) scores
	if model == "all":
		Utilities.getScores(PP_LMT_scores, PP_LML_scores, PP_indices)
	else:
		Utilities.getScoresofTransformer(PP_LMT_scores, PP_LML_scores, model, PP_indices)
	
	PP_R_scores = [[random.randint(0,1) for i in range(len(PP_H_scores[0]))] for j in range(20)] #random scores

	#lexical entailment (phase 2)
	LE_indices = [46, 36, 44, 47, 37, 45, 88, 38, 86, 89, 39, 87, 40, 90]
	LE_H_scores = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1], 
				[0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1], 
				[0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], 
				[0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1], 
				[0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], 
				[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1], 
				[1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1], 
				[1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], 
				[0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], 
				[0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], 
				[1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0], 
				[0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], 
				[0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], 
				[0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1], 
				[1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1], 
				[1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], 
				[0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1], 
				[1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1], 
				[1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1], 
				[0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1]]
	
	LE_LMT_scores = [] #LM (transformer-based) scores
	LE_LML_scores = [] #LM (LSTM-based) scores
	if model == "all":
		Utilities.getScores(LE_LMT_scores, LE_LML_scores, LE_indices)
	else:
		Utilities.getScoresofTransformer(LE_LMT_scores, LE_LML_scores, model, LE_indices)
	LE_R_scores = [[random.randint(0,1) for i in range(len(LE_H_scores[0]))] for j in range(20)] #random scores

	#quantifiers (phase 3)
	Q_indices = [68, 52, 58, 69, 61, 53, 59, 70, 494, 60, 71, 495, 505, 504, 809]
	Q_H_scores = [[1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0], 
				[1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0], 
				[1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0], 
				[1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0], 
				[1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0], 
				[1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0], 
				[1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0], 
				[1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0], 
				[1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0], 
				[1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0], 
				[1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], 
				[1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0], 
				[1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0], 
				[1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1], 
				[1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], 
				[1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0], 
				[1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0], 
				[1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0], 
				[1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0], 
				[1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0]]
	
	Q_LMT_scores = [] #LM (transformer-based) scores
	Q_LML_scores = [] #LM (LSTM-based) scores
	if model == "all":
		Utilities.getScores(Q_LMT_scores, Q_LML_scores, Q_indices)
	else:
		Utilities.getScoresofTransformer(Q_LMT_scores, Q_LML_scores, model, Q_indices)
	Q_R_scores = [[random.randint(0,1) for i in range(len(Q_H_scores[0]))] for j in range(20)] #random scores

	#propositional structure (phase 3)
	PS_indices = [2, 0, 354, 3, 1, 358, 355, 4, 367, 359, 5, 371, 362, 8, 429]
	PS_H_scores = [[0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0], 
				[0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0], 
				[0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0], 
				[0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0], 
				[0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0], 
				[0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0], 
				[0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1], 
				[0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0], 
				[0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0], 
				[0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0], 
				[0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0], 
				[0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0], 
				[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0], 
				[1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], 
				[0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0], 
				[0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0], 
				[1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0], 
				[0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0]]
	
	# PS_indices = [2,0,3,1,4,5,8] #subset of questions that are negation or double negation
	# toRemove = [14,12,11,9,8,6,5,2]
	# for i in toRemove:
	# 	for r in PS_H_scores:
	# 		r.pop(i)
	PS_LMT_scores = [] #LM (transformer-based) scores
	PS_LML_scores = [] #LM (LSTM-based) scores
	if model == "all":
		Utilities.getScores(PS_LMT_scores, PS_LML_scores, PS_indices)
	else:
		Utilities.getScoresofTransformer(PS_LMT_scores, PS_LML_scores, model, PS_indices)
	PS_R_scores = [[random.randint(0,1) for i in range(len(PS_H_scores[0]))] for j in range(20)] #random scores

	#richer logical structure (phase 4)
	RLS_indices = [347, 350, 346, 348, 351, 420, 349, 422, 421, 352, 423, 424, 353, 522, 425]
	RLS_H_scores = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], 
				[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1], 
				[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], 
				[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
	
	RLS_LMT_scores = [] #LM (transformer-based) scores
	RLS_LML_scores = [] #LM (LSTM-based) scores
	if model == "all":
		Utilities.getScores(RLS_LMT_scores, RLS_LML_scores, RLS_indices)
	else:
		Utilities.getScoresofTransformer(RLS_LMT_scores, RLS_LML_scores, model, RLS_indices)
	RLS_R_scores = [[random.randint(0,1) for i in range(len(RLS_H_scores[0]))] for j in range(20)] #random scores

	#world knowledge (phase 4)
	WK_indices = [565,688,564,566,689,580,567,692,581,590,693,671,591,704,674]
	WK_H_scores = [[1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0], 
				[1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], 
				[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1], 
				[0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0], 
				[0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0], 
				[1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1], 
				[0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1], 
				[1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0], 
				[0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1], 
				[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0], 
				[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0], 
				[1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1], 
				[1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1], 
				[1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0], 
				[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1], 
				[1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1], 
				[0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1], 
				[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1], 
				[1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1]]
	
	WK_LMT_scores = [] #LM (transformer-based) scores
	WK_LML_scores = [] #LM (LSTM-based) scores
	if model == "all":
		Utilities.getScores(WK_LMT_scores, WK_LML_scores, WK_indices)
	else:
		Utilities.getScoresofTransformer(WK_LMT_scores, WK_LML_scores, model, WK_indices)
	WK_R_scores = [[random.randint(0,1) for i in range(len(WK_H_scores[0]))] for j in range(20)] #random scores

	all_data = dict()
	all_data["Morphological Negation"] = [MN_indices, MN_H_scores, MN_LMT_scores, MN_LML_scores, MN_R_scores] #remove temporarily because variance is so low---almost everyone gets perfect scores
	all_data["Prepositional Phrases"] = [PP_indices, PP_H_scores, PP_LMT_scores, PP_LML_scores, PP_R_scores]
	all_data["Lexical Entailment"] = [LE_indices, LE_H_scores, LE_LMT_scores, LE_LML_scores, LE_R_scores]
	all_data["Quantifiers"] = [Q_indices, Q_H_scores, Q_LMT_scores, Q_LML_scores, Q_R_scores]
	all_data["Propositional Structure"] = [PS_indices, PS_H_scores, PS_LMT_scores, PS_LML_scores, PS_R_scores]
	all_data["Richer Logical Structure"] = [RLS_indices, RLS_H_scores, RLS_LMT_scores, RLS_LML_scores, RLS_R_scores]
	all_data["World Knowledge"] = [WK_indices, WK_H_scores, WK_LMT_scores, WK_LML_scores, WK_R_scores]

	all_indices = set()
	for subcat in all_data:
		all_indices = all_indices.union(all_data[subcat][0])
	all_indices = list(all_indices)

	if removePerfect:
		print("="*50, "\nREMOVING ALL INDIVIDUALS WITH PERFECT SCORES\n", "="*50)
		for subcat in all_data:
			D = all_data[subcat]
			for scores in D[1:5]:
				n = len(scores[0])
				# print("Before (perfect score", n, "):", [sum(I) for I in scores])
				while True:
					pCorrect = [sum(I) for I in scores]
					if n in pCorrect:
						scores.pop(pCorrect.index(n))
						continue
					else:
						break
				# input("After:"+str([sum(I) for I in scores]))

	if onlyClustering:
		ClusteringMethods.RunClustering(all_data, num_cluster_trials)
		print("\nAll Done!")
		exit()
	
	if ablation:
		Psychometrics.makeAblationHeatMap()
		print("\nAll Done!")
		exit()


	# for s in all_data:
	# 	print(s, ':', [len(X) for X in all_data[s][1:]])
	# exit()

	if doRanalysis:
		r = ro.r
		r['source']('IRTAnalysis.R')
		IRT = ro.globalenv['two_parameter']
		cats = ["Morphological Negation","Prepositional Phrases","Lexical Entailment","Quantifiers","Propositional Structure","Richer Logical Structure","World Knowledge"]
		runs = ['H','LMT','LML','R']
		for cat in cats:
			with open("R_outputs.txt","a") as f:
				f.write(f"\n{cat.upper()}")
				f.write("\n")
				f.write("="*50)
				f.write("\n")
			print(f"\n{cat}\n\n")
			for i in range(0,4):
				# convert to pandas dataframe
				print(f"{runs[i]}:\n")
				df = pd.DataFrame(all_data[cat][i+1])

				
				with localconverter(ro.default_converter + pandas2ri.converter):
					df_r = ro.conversion.py2rpy(df)
				# df_r = pandas2ri.py2ri(df)
				with open("R_outputs.txt","a") as f:
					f.write(f"{runs[i]}:\n")
				try:
					# some models may not due to exactly singular systems of linear equations
					# just skip those models
					IRT(df_r, cat, runs[i])
					time.sleep(2)
				except Exception:
					with open("R_outputs.txt","a") as f:
						f.write("\nERROR: Going to next step\n")
		print("\nAll Done!")
		exit()


	if onlyKeepTop:
		print("="*50, "\nUSING TOP N LANGUAGE MODELS ONLY\n", "="*50)
		for subcat in all_data:
			D = all_data[subcat]
			while len(D[2])>20:
				scores = [sum(I) for I in D[2]]
				i = scores.index(min(scores))
				D[2].pop(i)
	else:
		print("="*50, "\nUSING ALL LANGUAGE MODELS\n", "="*50)

	

	#plot the individual profiles
	for subcat in all_data:
		lms = ['', 'Human', 'LMT', 'LML', 'R']
		for i in range(1,5):
			D = all_data[subcat][i]
			plt.style.use('seaborn-darkgrid')
			for x1 in range(len(D[0])-1):
				x2 = x1+1
				#how many change from 0 to 1?
				widths = [0,0,0,0]
				for individual in D:
					# print("INDIVIDUAL:", individual)
					if individual[x1]==0:
						if individual[x2]==0:
							widths[0]+=1
						else:
							widths[1]+=1
					else:
						if individual[x2]==0:
							widths[2]+=1
						else:
							widths[3]+=1
				plt.plot([x1,x2], [0,0], marker='', color='green', linewidth=widths[0]/4, alpha=0.9)
				plt.plot([x1,x2], [0,1], marker='', color='green', linewidth=widths[1]/4, alpha=0.9)
				plt.plot([x1,x2], [1,0], marker='', color='green', linewidth=widths[2]/4, alpha=0.9)
				plt.plot([x1,x2], [1,1], marker='', color='green', linewidth=widths[3]/4, alpha=0.9)
				# print(widths)
			# plt.plot(x, y2, marker='', color='green', linewidth=1, alpha=0.9, label="l2")
			plt.title(subcat + ',' + lms[i])
			plt.savefig("individual_profiles.png")
	
	print("\nRaw Scores:")
	for subcat in all_data:
		D = all_data[subcat]
		print('\t', subcat, ':')
		for i in range(1,5):
			print('\t\t', mean([mean(I) for I in D[i]]))

	#this seems to predict the ability to estimate item difficulty the best
	print("\nScore variance by category (H, LMT, LML, R):")
	for subcat in all_data:
		D = all_data[subcat]
		print('\t', subcat, ':')
		for i in range(1,5):
			scores = [sum(individual)/len(individual) for individual in D[i]]
			print('\t\t', np.var(scores))


	print("\nAverage item variance by category (H,LMT,LML,R):")
	for subcat in all_data:
		D = all_data[subcat]
		print('\t', subcat, ':')
		for i in range(1,5):
			variances = [np.var([individual[j] for individual in D[i]]) for j in range(len(D[i][0]))]
			print('\t\t', mean(variances))

	print("\nCorrelation between human and machine item variances:")
	for subcat in all_data:
		D = all_data[subcat]
		print('\t', subcat, ':')
		H_variances = [np.var([individual[j] for individual in D[1]]) for j in range(len(D[1][0]))]
		LMT_variances = [np.var([individual[j] for individual in D[2]]) for j in range(len(D[2][0]))]
		LML_variances = [np.var([individual[j] for individual in D[3]]) for j in range(len(D[3][0]))]
		R_variances = [np.var([individual[j] for individual in D[4]]) for j in range(len(D[4][0]))]
		print('\t\tBetween H and LMT:', Utilities.pearsonr(H_variances, LMT_variances))
		print('\t\tBetween H and LML:', Utilities.pearsonr(H_variances, LML_variances))
		print('\t\tBetween H and R:', Utilities.pearsonr(H_variances, R_variances))
	
	if discrimination_index:
		print('\nDiscrimination Index per category (Correlation between H and LMT,LML,R):')
		for subcat in all_data:
			D = all_data[subcat]
			H_DIs = Psychometrics.getAllDiscriminationIndexes(D[1])
			print("\t", subcat, ':')
			print('\t\tBetween H and LMT:', Utilities.pearsonr(H_DIs, Psychometrics.getAllDiscriminationIndexes27(D[2])))
			print('\t\tBetween H and LML:', Utilities.pearsonr(H_DIs, Psychometrics.getAllDiscriminationIndexes27(D[3])))
			print('\t\tBetween H and R:', Utilities.pearsonr(H_DIs, Psychometrics.getAllDiscriminationIndexes27(D[4])))
	else:
		print("\nskipping...")
	
	print("\nSimple Difficulty:")
	for subcat in all_data:
		Psychometrics.outputDifficulty(subcat, all_data[subcat][0], model)
	
	print("\nItem-Total Correlation by Subcategory:")
	for subcat in all_data:
		D = all_data[subcat]
		Psychometrics.reportITC(subcat, D[1], D[2], D[3], D[4])
	
	print("\nInter-Item Correlation by Subcategory:")
	for subcat in all_data:
		D = all_data[subcat]
		Psychometrics.reportIIC(subcat, D[1], D[2], D[3], D[4])
	
	# Across-category average IIC:
    # * take 5000 random subsets of questions from *all subcategories* of size 15.
    # * Calculate AIIC for each of these new subsets using human, LM data. Then see whether they correlate across the 5000 subsets.
	print("\n*constructing random_subsets...")
	random_subsets = [random.sample(all_indices,5) for x in range(100)] #1000

	random_pairs = [random.sample(all_indices,2) for x in range(100)]
	H_AIICs,LMT_AIICs,LML_AIICs,LM_AIICs,R_AIICs = [[],[],[],[],[]]
	for indices in random_pairs:
		H_AIICs.append( mean(Psychometrics.calcAllIIC( [allScoresFromQuestionId[Id] for Id in indices] )) )
		LMT_AIIC_scores,LML_AIIC_scores = [[],[]]
		Utilities.getScores(LMT_AIIC_scores, LML_AIIC_scores, indices)
		LMT_AIICs.append( mean(Psychometrics.calcAllIIC( LMT_AIIC_scores )) )
		LML_AIICs.append( mean(Psychometrics.calcAllIIC( LML_AIIC_scores )) )
		LM_AIICs.append( mean(Psychometrics.calcAllIIC( LMT_AIIC_scores + LML_AIIC_scores )) )
		R_AIICs.append( mean(Psychometrics.calcAllIIC( [[random.randint(0,1) for i in range(2)] for j in range(20)] )) )

	print('\nIIC across', len(random_pairs), 'random pairs of items (pearson corr):')
	print('\tFrom H to LMT:', Utilities.pearsonr(H_AIICs, LMT_AIICs), Utilities.MSE(H_AIICs, LMT_AIICs))
	print('\tFrom H to LML:', Utilities.pearsonr(H_AIICs, LML_AIICs), Utilities.MSE(H_AIICs, LML_AIICs))
	print('\tFrom H to LM:', Utilities.pearsonr(H_AIICs, LM_AIICs), Utilities.MSE(H_AIICs, LM_AIICs))
	print('\tFrom H to R:', Utilities.pearsonr(H_AIICs, R_AIICs), Utilities.MSE(H_AIICs, R_AIICs))


	print('\nAIIC per category (H,LMT,LML,LM,R):')
	for subcat in all_data:
		D = all_data[subcat]
		results = [str(mean(Psychometrics.calcAllIIC(s))) for s in [D[1], D[2], D[3], D[2]+D[3], D[4]]]
		print('\t', subcat, ':\n\t\t', '\n\t\t'.join(results) )



	# ANALYZE CRONBACH'S ALPHA:
	# * take 5000 random subsets of questions from *all subcategories* of size 15.
	# * For each subset of questions, calculate alpha for LM responses, and people responses
	# * Determine correlation between LM/people alpha scores over all subsets.
	# * Determine mean squared distance between LM/people alpha scores over all subsets.
	H_CAs,LMT_CAs,LML_CAs,LM_CAs,R_CAs = [[],[],[],[],[]]
	print("\n*calculating Cronbach's alphas...")
	for (prog,indices) in enumerate(random_subsets):
		if prog%250==0:
			print('*\t',prog,"of",len(random_subsets))
		H_CAs.append(Psychometrics.calcCronbachAlpha( [allScoresFromQuestionId[Id] for Id in indices] ))
		LMT_scores,LML_scores = [[],[]]
		Utilities.getScores(LMT_scores, LML_scores, indices)
		LMT_CAs.append( Psychometrics.calcCronbachAlpha( LMT_scores ) )
		LML_CAs.append( Psychometrics.calcCronbachAlpha( LML_scores ) )
		LM_CAs.append( Psychometrics.calcCronbachAlpha( LMT_scores + LML_scores ) )
		R_CAs.append( Psychometrics.calcCronbachAlpha( [[random.randint(0,1) for i in range(15)] for j in range(20)] ) )
		removeLast = False
		for L in [H_CAs, LMT_CAs, LML_CAs, LM_CAs, R_CAs]:
			if math.isnan(L[-1]) or math.isinf(L[-1]):
				removeLast = True
		if removeLast:
			#remove this one from all of them
			for L in [H_CAs, LMT_CAs, LML_CAs, LM_CAs, R_CAs]:
				L.pop(-1)


	print("\nCronbach's alpha across", len(random_subsets), "randomized subsets of questions (pearson corr. and MSE):")
	print('\tFrom H to LMT:', Utilities.pearsonr(H_CAs, LMT_CAs), Utilities.spearmanr(H_CAs, LMT_CAs), Utilities.MSE(H_CAs, LMT_CAs))
	print('\tFrom H to LML:', Utilities.pearsonr(H_CAs, LML_CAs), Utilities.spearmanr(H_CAs, LML_CAs), Utilities.MSE(H_CAs, LML_CAs))
	print('\tFrom H to LM:', Utilities.pearsonr(H_CAs, LM_CAs), Utilities.spearmanr(H_CAs, LM_CAs), Utilities.MSE(H_CAs, LM_CAs))
	print('\tFrom H to R:', Utilities.pearsonr(H_CAs, R_CAs), Utilities.spearmanr(H_CAs, R_CAs), Utilities.MSE(H_CAs, R_CAs))

	print('\nCronbachs Alpha per category (H,LMT,LML,LM,R):')
	for subcat in all_data:
		D = all_data[subcat]
		results = [str(Psychometrics.calcCronbachAlpha(s)) for s in [D[1], D[2], D[3], D[2]+D[3], D[4]]]
		print('\t', subcat, ':\n\t\t', '\n\t\t'.join(results) )

	print('\nGreedy Item Removal, AIIC, AITC, and Cronbachs after reduction to 5 items (H,LMT,LML,R)')
	print('(Scores are percentages of the scores obtained through human responses):')
	for subcat in all_data:
		D = all_data[subcat]
		H_results = Psychometrics.performItemRemoval_greedy(D[1], D[1])
		print('\t', subcat, ':')
		for i in range(1,5):
			results = Psychometrics.performItemRemoval_greedy(D[i], D[1])
			print('\t\t', [results[j]/H_results[j] for j in range(3)])
			# print(H_results, '\n', Psychometrics.performItemRemoval_greedy(D[0], D[1], D[1]))

	print('\nGlobal Item Removal, AIIC, AITC, and Cronbachs after reduction to 5 items (LMT,LML,R) using metric', metric)
	print('(Scores are percentages of the scores obtained through human responses):')
	for subcat in all_data:
		D = all_data[subcat]
		H_result = Psychometrics.performItemRemoval_global(D[1], D[1], metric=metric)
		print('\t', subcat, ':')
		for i in range(2,5):
			result = Psychometrics.performItemRemoval_global(D[i], D[1], metric=metric)
			print('\t\t', result/H_result)
			# print(H_results, '\n', Psychometrics.performItemRemoval_greedy(D[0], D[1], D[1]))

	print("\nAll Done!")
