from util import *

# Add your import statements here




class InflectionReduction:

	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""

		reducedText = None

		#Fill in code here
		if not isinstance(text, list):
			print("Missing Text")
			return None

		porterStemmer = PorterStemmer()
		
		reducedText =[[porterStemmer.stem(word) for word in sentence if isinstance(word, str)] for sentence in text if isinstance(sentence, list)]
		return reducedText


