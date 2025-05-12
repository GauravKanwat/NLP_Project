from util import *

# Add your import statements here
stop_words = set(stopwords.words('english'))

class StopwordRemoval():

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		stopwordRemovedText = None

		#Fill in code here
		filtered_sentences = []
		for sentence in text:
			filtered_sentence = [word for word in sentence if word.lower() not in stop_words]
			filtered_sentences.append(filtered_sentence)
		stopwordRemovedText=filtered_sentences
		return stopwordRemovedText


	def probabilisticMethod(self, sentences):
		"""
		Probabilistic Frequency-based Stopword Removal

		Parameters
		----------
		sentences : list
			A list of lists where each sub-list is a sequence of tokens representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens representing a sentence with stopwords removed
		"""
		probability_threshold=0.15
		# Calculate word frequencies and total number of words
		token_freq, total_tokens = self._calculate_frequencies(sentences)
		# Calculate token probabilities
		token_prob = self._calculate_probabilities(token_freq, total_tokens)
		# Identify stopwords based on the probability threshold
		frequent_tokens = self._identify_stopwords(token_prob, probability_threshold)
		# Remove stopwords from each sentence
		cleaned_sentences = self._remove_stopwords(sentences, frequent_tokens)

		return cleaned_sentences

	def _calculate_frequencies(self, sentences):
		all_tokens = [token.lower() for sentence in sentences for token in sentence]
		token_freq = Counter(all_tokens)
		total_tokens = sum(token_freq.values())
		return token_freq, total_tokens

	def _calculate_probabilities(self, token_freq, total_tokens):
		return {token: freq / total_tokens for token, freq in token_freq.items()}

	def _identify_stopwords(self, token_prob, probability_threshold):
		return {token for token, prob in token_prob.items() if prob > probability_threshold}

	def _remove_stopwords(self, sentences, stopwords):
		return [
			[token for token in sentence if token.lower() not in stopwords]
				for sentence in sentences]


	