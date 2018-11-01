from nltk.tokenize import sent_tokenize, word_tokenize

text = 'Bakan Kurum, "Bir zaman ülkemizde yüksek binalar gerçekten ihtiyaçtı. Hem imar planlarının azlığı, hem nüfusun yoğunluğu, kentlere göçün hızı bir şekilde olması yüksek bina ihtiyacını beraberinde getirdi. Ancak bugün yüksek binalar bizleri gerçekten rahatsız ediyor. 30-40 katlı binaları kentsel dönüşümde de görüyoruz.'
text = word_tokenize(text)

from nltk.corpus import stopwords
stopwords = stopwords.words('turkish')

filtered_words = []
for word in text:
    if word not in stopwords:
        filtered_words.append(word)

from nltk.stem import PorterStemmer
ps = PorterStemmer()

words = ['drive', 'driving', 'driver', 'cats']
for w in words:
    ps.stem(w)
    
import nltk
text = 'This article in a continuation in my series of articles aimed at ‘Explainable Artificial Intelligence (XAI)’. If you haven’t checked out the first article, I would definitely recommend you to take a quick glance at ‘Part I — The Importance of Human Interpretable Machine Learning’ which covers the what and why of human interpretable machine learning and the need and importance of model interpretation along with its scope and criteria. In this article, we will be picking up from where we left off and expand further into the criteria of machine learning model interpretation methods and explore techniques for interpretation based on scope. The aim of this article is to give you a good understanding of existing, traditional model interpretation methods, their limitations and challenges. We will also cover the classic model accuracy vs. model interpretability trade-off and finally take a look at the major strategies for model interpretation.'
tokenized = word_tokenize(text)
tag = nltk.pos_tag(tokenized)

text = '''Our Apple history feature includes information about The foundation of Apple and the years that followed, we look at How Jobs met Woz and Why Apple was named Apple. The Apple I and The debut of the Apple II. Apple's visit to Xerox, and the one-button mouse. The story of The Lisa versus the Macintosh. Apple's '1984' advert, directed by Ridley Scott. The Macintosh and the DTP revolution. We go on to examine what happened between Jobs and Sculley, leading to Jobs departure from Apple, and what happened during The wilderness years: when Steve Jobs wasn't at Apple, including Apple's decline and IBM and Microsoft's rise and how Apple teamed up with IBM and Motorola and eventually Microsoft. And finally, The return of Jobs to Apple.'''
tokenized = word_tokenize(text)
tag = nltk.pos_tag(tokenized)
named_ent = nltk.ne_chunk(tag)
named_ent.draw()

from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()
words = ['drive', 'driving', 'driver', 'cats']
for w in words:
    print(lem.lemmatize(w))