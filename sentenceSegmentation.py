from util import *

# Add your import statements here


class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = None

		#Fill in code here
		if not isinstance(text, str):
			print("Missing Text")
			return []

		segments = re.split("[.,?!]", text)
		segmentedText = [s.strip() for s in segments if s.strip()]
		return segmentedText





	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""

		segmentedText = None

		#Fill in code here
		if not isinstance(text, str):
			print("Missing Text")
			return []

		tokenizer = PunktSentenceTokenizer(text)
		segmentedText=tokenizer.tokenize(text)
		return segmentedText