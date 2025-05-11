from util import *

# Add your import statements here




class InformationRetrieval():

	def __init__(self):
		self.index = None
		self.types = None
		self.docIDs = None
		self.document_term_freq = None


	def buildIndex(self, docs, docIDs):
		"""
		Constructs an inverted index and supporting structures for a given set of documents.

		Parameters
		----------
		docs : list
			A list of documents, where each document is a list of sentences and each sentence is a list of terms.
		docIDs : list
			A list of integers representing the document IDs.

		Returns
		-------
		None
		"""

		# Initialize required data structures
		inverted_index = {}
		all_terms = set()
		doc_term_frequencies = {}

		merged_documents = []

		# Process each document and its corresponding ID
		for doc_idx, (doc, doc_id) in enumerate(zip(docs, docIDs)):
			flat_doc = []
			term_counts = {}

			# Flatten sentences and compute term frequencies
			for sentence in doc:
				for word in sentence:
					flat_doc.append(word)
					all_terms.add(word)

					# Update frequency count
					term_counts[word] = term_counts.get(word, 0) + 1

					# Update inverted index
					if word not in inverted_index:
						inverted_index[word] = [doc_id]
					elif doc_id not in inverted_index[word]:
						inverted_index[word].append(doc_id)

			# Store the merged document and term frequencies
			merged_documents.append(flat_doc)
			doc_term_frequencies[doc_idx] = term_counts

		# Store the processed data in instance variables
		self.index = inverted_index
		self.types = list(all_terms)
		self.docIDs = docIDs
		self.document_term_freq = doc_term_frequencies

	
	def compute_tfidf_matrix(self):
		"""
		Computes the TF-IDF matrix for a collection of documents.

		Returns
		-------
		np.ndarray
			A 2D numpy array representing the TF-IDF scores.
		"""
		total_docs = len(self.docIDs)
		total_terms = len(self.types)

		# Create an empty matrix for TF-IDF scores
		tfidf = np.zeros((total_docs, total_terms))

		# Precompute document frequencies for each term
		doc_freq = {term: len(doc_list) for term, doc_list in self.index.items()}

		# Map term to its column index for quick lookup
		term_index_map = {term: idx for idx, term in enumerate(self.index)}

		# Fill TF-IDF matrix row-wise (document-wise)
		for doc_idx, doc_id in enumerate(self.docIDs):
			term_freqs = self.document_term_freq[doc_id - 1]
			for term, freq in term_freqs.items():
				if term in term_index_map:
					col_idx = term_index_map[term]
					idf = math.log10(total_docs / doc_freq[term])
					tfidf[doc_idx][col_idx] = freq * idf

		return tfidf

	def compute_query_tfidf(self, num_queries, query_inverted_index, query_term_frequencies):
		"""
		Compute the TF-IDF matrix for a set of queries using the main document collection's statistics.

		Parameters
		----------
		num_queries : int
			The total number of queries.
		query_inverted_index : dict
			An inverted index for the queries mapping terms to lists of query IDs that contain the term.
		query_term_frequencies : dict
			A dictionary mapping each query ID to another dictionary of term frequencies.

		Returns
		-------
		np.ndarray
			A 2D numpy array where each row corresponds to a query and each column to a term.
		"""

		total_docs = len(self.docIDs)       # Total number of documents in the collection
		total_terms = len(self.types)       # Total number of unique terms in the collection

		# Initialize an empty TF-IDF matrix for the queries
		tfidf_queries = np.zeros((num_queries, total_terms))

		# Create a mapping from term to its column index in the matrix
		term_to_col = {term: idx for idx, term in enumerate(self.index)}

		# Precompute document frequency (df) for each term from the document collection
		doc_freq = {term: len(self.index[term]) for term in self.index}

		# Iterate over each term in the query inverted index
		for term, query_ids in query_inverted_index.items():
			# Check if the term exists in the document index (to compute IDF)
			if term in term_to_col and term in doc_freq:
				idf = math.log10(total_docs / doc_freq[term])  # Compute IDF for the term
				col_idx = term_to_col[term]                    # Get the column index for this term

				# For each query that contains the term
				for qid in query_ids:
					tf = query_term_frequencies[qid].get(term, 0)  # Get term frequency in the query
					tfidf_queries[qid][col_idx] = tf * idf         # Calculate and store TF-IDF score

		return tfidf_queries
	
	def query_inv_index_maker(self, queries):
		"""
		Builds an inverted index for a list of queries and computes term frequencies for each.

		Parameters
		----------
		queries : list
			A list where each element is a query, represented as a list of sentences (each sentence is a list of words).

		Returns
		-------
		tuple
			- Inverted index of query terms: dict mapping term → list of query IDs
			- Term frequencies per query: dict mapping query ID → {term: count}
			- List of merged queries
		"""

		merged_queries_list = []
		inverted_index = {}
		term_frequencies = {}

		# Prepare empty lists in the inverted index for all terms from the document index
		for term in self.index:
			inverted_index[term] = []

		# Process each query
		for qid, query in enumerate(queries):
			flattened_query = []
			freq_counter = {}

			# Flatten sentences and count term frequencies
			for sentence in query:
				for word in sentence:
					flattened_query.append(word)

					if word in self.index:
						# Count term frequency
						freq_counter[word] = freq_counter.get(word, 0) + 1

						# Update inverted index if not already added
						if qid not in inverted_index[word]:
							inverted_index[word].append(qid)

			# Ensure all terms from document index are included in frequency dictionary (even if 0)
			for term in self.index:
				freq_counter.setdefault(term, 0)

			# Save results
			merged_queries_list.append(flattened_query)
			term_frequencies[qid] = freq_counter

		return inverted_index, term_frequencies, merged_queries_list


	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		# Get TF-IDF matrix for documents
		doc_tfidf = self.compute_tfidf_matrix()

		# Generate query inverted index, term frequencies, and flattened queries
		q_inv_idx, query_term_freq, merged_queries = self.query_inv_index_maker(queries)

		# Create TF-IDF matrix for queries
		query_tfidf = self.compute_query_tfidf(len(merged_queries), q_inv_idx, query_term_freq)

		doc_IDs_ordered = []  # To store ranked document IDs for each query

		for q_idx, q_vec in enumerate(query_tfidf):
			similarity_scores = {}

			for d_idx, d_vec in enumerate(doc_tfidf):
				dot_product = np.dot(d_vec, q_vec)
				doc_norm = np.linalg.norm(d_vec)
				query_norm = np.linalg.norm(q_vec)

				if not (dot_product and doc_norm and query_norm):
					similarity = 0.0
				else:
					similarity = dot_product / (doc_norm * query_norm)

				similarity_scores[self.docIDs[d_idx]] = similarity

			# Sort document IDs by similarity score (descending)
			sorted_docs = sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)
			ranked_ids = [doc_id for doc_id, _ in sorted_docs]
			doc_IDs_ordered.append(ranked_ids)

		return doc_IDs_ordered



	def create_svd_index(self, documents, doc_ids):
		"""
		Builds a reduced-dimensionality document index using TF-IDF and Truncated SVD.

		Parameters
		----------
		documents : list of list of list of str
			A list where each document is represented as a list of sentences,
			and each sentence is a list of words.
		doc_ids : list of int
			Unique identifiers for the documents.

		Returns
		-------
		None
		"""

		# Step 1: Flatten each document into a plain string
		processed_docs = []
		for doc in documents:
			flat_doc = ' '.join(' '.join(sentence) for sentence in doc)
			processed_docs.append(flat_doc)

		# Step 2: Vectorize the corpus using TF-IDF
		tfidf_vectorizer = TfidfVectorizer()
		tfidf_matrix = tfidf_vectorizer.fit_transform(processed_docs)

		# Step 3: Normalize the TF-IDF matrix
		normalized_matrix = normalize(tfidf_matrix, norm='l2', axis=1)

		# Step 4: Apply Truncated SVD for dimensionality reduction
		svd = TruncatedSVD(n_components=500, random_state=42)
		reduced_matrix = svd.fit_transform(normalized_matrix)

		# Step 5: Save results to instance variables
		self.index = reduced_matrix
		self.vectorizer = tfidf_vectorizer
		self.svd_model = svd
		self.num_features = tfidf_matrix.shape[1]


	def svd_rank(self, query_list):
		"""
		Ranks documents by relevance for each query using the SVD-reduced document index.

		Parameters
		----------
		query_list : list
			A list where each query is a list of sentences,
			and each sentence is a list of words.

		Returns
		-------
		list
			A list of lists of integers. Each inner list contains document IDs 
			ranked by predicted relevance for the corresponding query.
		"""

		all_ranked_results = []

		for query in query_list:
			# Flatten query into a single string
			flat_query = ' '.join(' '.join(sentence) for sentence in query)

			# Vectorize query using trained TF-IDF vectorizer
			query_vector_tfidf = self.vectorizer.transform([flat_query])

			# Project query vector into the same reduced space as documents
			query_vector_svd = self.svd_model.transform(query_vector_tfidf)

			# Compute cosine similarity between query and all documents
			similarity_scores = cosine_similarity(query_vector_svd, self.index)

			# Rank document indices by descending similarity
			ranked_indices = np.argsort(similarity_scores[0])[::-1]

			# Adjust to 1-based indexing (if required)
			ranked_doc_ids = (ranked_indices + 1).tolist()

			all_ranked_results.append(ranked_doc_ids)

		return all_ranked_results


	def create_esa_index(self, docs, doc_ids):
		"""
		Build ESA-like concept space index using gensim LSI model on a simulated Wikipedia concept space.

		Parameters
		----------
		docs : list
			A list where each document is a list of sentences,
			and each sentence is a list of words.

		Returns
		-------
		None
		"""
		# Flatten and preprocess documents
		flattened_docs = [' '.join([' '.join(sentence) for sentence in doc]) for doc in docs]
		tokenized_docs = [doc.lower().split() for doc in flattened_docs]

		# Create dictionary and corpus
		self.esa_dictionary = corpora.Dictionary(tokenized_docs)
		corpus = [self.esa_dictionary.doc2bow(text) for text in tokenized_docs]

		# Create TF-IDF model
		tfidf_model = models.TfidfModel(corpus)

		# Transform corpus to TF-IDF
		tfidf_corpus = tfidf_model[corpus]

		# Create LSI model (approximating concept space)
		self.esa_lsi_model = models.LsiModel(tfidf_corpus, id2word=self.esa_dictionary, num_topics=300)

		# Transform corpus into LSI space
		lsi_corpus = self.esa_lsi_model[tfidf_corpus]

		# Build similarity index
		self.esa_index = similarities.MatrixSimilarity(lsi_corpus, num_features=300)


	def esa_rank(self, queries):
		"""
		Rank documents using ESA-based vector representation and similarity.

		Parameters
		----------
		queries : list
			A list of queries, each query is a list of sentences,
			and each sentence is a list of words.

		Returns
		-------
		list
			A list of lists containing document IDs ranked by ESA similarity.
		"""

		ranked_doc_ids_all = []

		for query in queries:
			flat_query = ' '.join([' '.join(sentence) for sentence in query])
			query_bow = self.esa_dictionary.doc2bow(flat_query.lower().split())
			query_lsi = self.esa_lsi_model[query_bow]

			similarities_scores = self.esa_index[query_lsi]
			ranked_indices = np.argsort(similarities_scores)[::-1]
			ranked_ids = (ranked_indices + 1).tolist()  # +1 to match 1-based indexing
			ranked_doc_ids_all.append(ranked_ids)

		return ranked_doc_ids_all
