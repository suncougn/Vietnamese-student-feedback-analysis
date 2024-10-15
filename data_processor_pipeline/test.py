from underthesea import word_tokenize

with open('vietnamese-stopwords.txt', 'r', encoding='utf-8') as f:
	stws=f.readlines()
stwsv2=[stw.strip() for stw in stws]

strings="Thầy dạy rất nhiệt tình nhưng mà nhiều học sinh không chú ý lắng nghe"
strings=word_tokenize(strings, format='text')

def remove_vn_stopwords(strings):
	strings=strings.split()
	doc_words=[word for word in strings if word not in stwsv2]
	print(doc_words)

remove_vn_stopwords(strings)