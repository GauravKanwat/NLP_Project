from util import *

# Add your import statements here




class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		precision = 0

		# Get the top-k retrieved document IDs
		top_k_docs = query_doc_IDs_ordered[:k]

		# Count how many of the top-k documents are actually relevant
		relevant_hits = sum(1 for doc_id in top_k_docs if doc_id in true_doc_IDs)

		# Handle division by zero if k is zero
		if k == 0:
			return 0.0

		# Compute precision
		precision = relevant_hits / k

		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		meanPrecision = 0

		#Fill in code here
		total_precision = 0.0  # Cumulative sum of precision values

		for idx, query_id in enumerate(query_ids):
			query_doc_IDs = doc_IDs_ordered[idx]

			# Extract relevant documents for this query
			relevant_docs = [
				int(entry['id']) for entry in qrels 
				if int(entry['query_num']) == query_id
			]

			# Compute precision for current query and accumulate
			total_precision += self.queryPrecision(query_doc_IDs, query_id, relevant_docs, k)

		# Compute mean precision over all queries
		meanPrecision = total_precision / len(query_ids) if query_ids else 0.0



		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		recall = 0

		#Fill in code here
		total_relevant = len(true_doc_IDs)
		if total_relevant == 0:
			return 0.0

		# Use set intersection to find relevant documents retrieved in top-k
		top_k_results = set(query_doc_IDs_ordered[:k])
		relevant_found = len(top_k_results.intersection(true_doc_IDs))

		recall = relevant_found / total_relevant

		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		meanRecall = 0

		#Fill in code here
		recall_scores = []

		# Create a mapping from query ID to relevant document IDs
		query_relevance_map = {}
		for entry in qrels:
			q_id = int(entry['query_num'])
			doc_id = int(entry['id'])
			if q_id not in query_relevance_map:
				query_relevance_map[q_id] = []
			query_relevance_map[q_id].append(doc_id)

		# Compute recall per query and collect the results
		for idx, query_id in enumerate(query_ids):
			retrieved_docs = doc_IDs_ordered[idx]
			relevant_docs = query_relevance_map.get(query_id, [])  # Slice to top-k if needed

			if relevant_docs:
				recall = self.queryRecall(retrieved_docs, query_id, relevant_docs, k)
				recall_scores.append(recall)
			else:
				recall_scores.append(0.0)

		# Return average recall
		meanRecall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0

		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = 0

		beta = 0.5
		beta_sq = beta ** 2

		# Get precision and recall for this query
		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)

		# Calculate F-beta score
		denominator = beta_sq * precision + recall
		if denominator > 0:
			fscore = (1 + beta_sq) * precision * recall / denominator
		else:
			fscore = 0.0

		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = 0

		total_fscore = 0.0
		query_count = len(query_ids)

		for idx, query_id in enumerate(query_ids):
			predicted_docs = doc_IDs_ordered[idx]

			relevant_docs = [
				int(qrel['id']) for qrel in qrels
				if int(qrel['query_num']) == query_id
			]

			fscore = self.queryFscore(predicted_docs, query_id, relevant_docs, k)
			total_fscore += fscore

		meanFscore = total_fscore / query_count if query_count > 0 else 0.0

		return meanFscore
	
	'''
	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		nDCG = 0

		# Relevance assignment: 5 for highest, decreasing based on order in true_doc_IDs
		rel_grades = {
			doc_id: 4 - idx for idx, doc_id in enumerate(true_doc_IDs)
		}

		# Compute DCG from predicted order
		DCG = 0.0
		for rank, doc_id in enumerate(query_doc_IDs_ordered[:k]):
			score = rel_grades.get(doc_id, 0)
			DCG += score / math.log2(rank + 2)

		# Compute IDCG from ideal ranking
		ideal_scores = sorted([rel_grades.get(doc_id, 0) for doc_id in query_doc_IDs_ordered[:k]], reverse=True)
		IDCG = sum(score / math.log2(i + 2) for i, score in enumerate(ideal_scores))
		nDCG=DCG / IDCG if IDCG > 0 else 0.0

		return nDCG
	'''

	def collect_relevant_docs(self, query_doc_IDs_ordered, true_doc_IDs, k):
		# Collect relevant documents up to rank k
		relevant_docs = []
		for doc_id in query_doc_IDs_ordered[:k]:
			if doc_id in true_doc_IDs:
				relevant_docs.append(doc_id)
		return relevant_docs
	
	def compute_relevance_score(self, query_doc_IDs_ordered, relevant_docs, k):
		# Compute relevance scores for each document
		relevance_scores = []
		for doc_id in query_doc_IDs_ordered[:k]:
			relevance_score = 0  # Default relevance score is 0

			# Graded relevance (or can be made 4 to 1)
			if doc_id in relevant_docs:
				relevance_score = 5 - relevant_docs.index(doc_id)  # Compute relevance score for the document
			relevance_scores.append(relevance_score)  # Append relevance score to the list
		return relevance_scores

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""


		nDCG = 0  # Initialize nDCG.

		relevant_docs = self.collect_relevant_docs(query_doc_IDs_ordered,true_doc_IDs,k)
		

		relevance_scores = self.compute_relevance_score(query_doc_IDs_ordered, relevant_docs, k)
		

		# Compute ideal relevance scores
		sorted_relevance_scores = sorted(relevance_scores, reverse=True)

		# Compute DCG
		DCG = sum([relevance_scores[i] / math.log2(i + 2) for i in range(min(k, len(query_doc_IDs_ordered)))])

		# Compute ideal DCG
		IDCG = sum([sorted_relevance_scores[i] / math.log2(i + 2) for i in range(min(k, len(query_doc_IDs_ordered)))])

		# Compute nDCG
		nDCG = DCG / IDCG if IDCG > 0 else 0  # Avoid division by zero


		return nDCG

	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		meanNDCG = 0

		total_ndcg = 0.0  # Accumulator for all nDCG values

		# Evaluate nDCG for each query
		for i in range(len(query_ids)):
			query_id = query_ids[i]
			query_doc_IDs = doc_IDs_ordered[i]

			# Gather relevant documents for the current query
			true_doc_IDs = [
				int(qrel['id']) for qrel in qrels if int(qrel['query_num']) == query_id
			]

			# Compute nDCG for the current query and add it to the total
			ndcg = self.queryNDCG(query_doc_IDs, query_id, true_doc_IDs, k)
			total_ndcg += ndcg

		meanNDCG = total_ndcg / len(query_ids) if query_ids else 0.0
		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		avgPrecision = 0.0  # Initialize the return value

		sum_precisions = 0.0
		relevant_found = 0

		# Evaluate documents in the predicted order
		for i, doc_id in enumerate(query_doc_IDs_ordered):
			if doc_id in true_doc_IDs:
				relevant_found += 1
				sum_precisions += relevant_found / (i + 1)
			if relevant_found >= k:
				break

		# Final computation only if relevant documents were found
		if relevant_found > 0:
			avgPrecision = sum_precisions / relevant_found

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		meanAveragePrecision = 0

		total_avg_precision = 0.0  # Sum of average precision scores over all queries

		for i in range(len(query_ids)):
			query_id = query_ids[i]
			query_doc_IDs_ordered = doc_IDs_ordered[i]

			# Identify relevant documents for the current query from qrels
			true_doc_IDs = []
			for qrel in q_rels:
				if int(qrel['query_num']) == query_id:
					true_doc_IDs.append(int(qrel['id']))

			# Compute average precision for this query and accumulate it
			avg_precision = self.queryAveragePrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
			total_avg_precision += avg_precision

		meanAveragePrecision = total_avg_precision / len(query_ids) if query_ids else 0.0
		return meanAveragePrecision

