from util import *

# Add your import statements here




class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = None

		#Fill in code here
		if not isinstance(text, list):
			print("Missing Text")
			return []

		text_separators = r"\s+"  # Example: Split by spaces
		punctuation_pattern = r"[.,!?;:\"'(){}[\]<>]"  # Regex to remove punctuation

		tokenizedText = [
			[re.sub(punctuation_pattern, "", token) for token in re.split(text_separators, sentence) if token]
			for sentence in text if isinstance(sentence, str)
		]

		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = None

		#Fill in code here
		if not isinstance(text, list):
			print("Missing Text")
			return []

		tokenizer = TreebankWordTokenizer()

		tokenizedText=[tokenizer.tokenize(sentence) for sentence in text if isinstance(sentence, str)]
		return tokenizedText