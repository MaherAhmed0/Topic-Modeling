import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import wordcloud
import gensim
from gensim.models.ldamodel import LdaModel
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from pprint import pprint
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords

##### Loading Data set #####
MainData = pd.read_csv('articles1.csv')
pd.set_option('max_columns', None)

print("_________________________________________________")
print("Loading data done.")
print("_________________________________________________")
####################################

##### Preprocessing Data set #####
# Index column in original dataset we renamed it then dropped it #
MainData.rename(columns={'Unnamed: 0': 'Name'}, inplace=True)
# dropping all columns that doesn't affect the topic modeling #
MainData = MainData.drop(columns=['Name', 'id', 'title', 'author', 'publication', 'date', 'year', 'month', 'url']
                         , axis=1)

### Model_1 Part ###


def remove_punc(data):
    # Function to remove punctuation from data #
    punc = '''!()-[]{};:'",<>./?@#$%^&*_~'''
    for ele in data:
        if ele in punc:
            data = data.replace(ele, "")
    return data


def convert_lower(data):
    # Function to convert all upper case letters to lower case #
    for column in data.columns:
        data[column] = data[column].str.lower()
    return data


# Calling the functions to apply preprocessing on the dataset #
MainData = remove_punc(MainData)
MainData = convert_lower(MainData)

stop_words = stopwords.words("english")
# Count Vectorizer tokenizes the text alongside with removing stop words #
CountVec = CountVectorizer(stop_words=stop_words)

# Convert the matrix to numerical matrix #
Data = CountVec.fit_transform(MainData.content)
### Model_2 Part ###


def remove_stopwords(texts):
    # Function to remove stop words from data #
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]


def sent_to_words(sentences):
    for sentence in sentences:
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)
        # deacc parameter is for removing accent marks #


# convert data to list #
Model_2_Data = MainData.content.tolist()
# Calling the functions to apply preprocessing on the dataset #
Data_Words = list(sent_to_words(Model_2_Data))
Data_Words = remove_stopwords(Data_Words)
print("Tokenized words sample:", Data_Words[:1][0][:50])

Dict = corpora.Dictionary(Data_Words)

Corpus = [Dict.doc2bow(text) for text in Data_Words]
print("numerical representation (bag of words) sample:", Corpus[:1][0][:50])

print("_________________________________________________")
print("Preprocessing done.")
print("_________________________________________________")
#####################################

##### Data Visualization #####
# Word cloud generates an image with the most common words in the given data #
Words = " ".join(MainData.content)
wordcloud = wordcloud.WordCloud()
wordcloud.generate(Words)
wordcloud.to_image().show()


def plot_most_common_words(data_count, countvec):
    # Function to plot a graph that shows the count of the most common words #
    words = countvec.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in data_count:
        total_counts += t.toarray()[0]

    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words))
    plt.bar(x_pos, counts, align='center')
    plt.xticks(x_pos, words, rotation=90)
    plt.xlabel('Words')
    plt.ylabel('Count')
    plt.title('Most common words')
    plt.show()


plot_most_common_words(Data, CountVec)
print("Visualization done.")
print("_________________________________________________")
#####################################

##### Model_1 Training #####
Train_Data, Test_Data = train_test_split(Data, test_size=0.20, random_state=0)

number_topics = 10
number_words = 10
# sklearn model #
LDA = LatentDirichletAllocation(n_components=number_topics, random_state=0)
LDA.fit(Train_Data)

print("Model_1 Training done.")
print("_________________________________________________")
##### Model Evaluation and Results #####


def print_topics(lda_model, feature_names, n_top_words):
    # Function to print each topic alongside with its most common words #
    words = feature_names.get_feature_names()

    topic_words = lda_model.components_
    for topic_idx, topic in enumerate(topic_words):

        print("\nMost common words in topic %d:" % (topic_idx + 1))
        # sort top words according to their value
        print(", ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))


# # Calling the function to print #
print("Topics found via Sklearn LDA:")
print_topics(LDA, CountVec, number_words)
print("_________________________________________________")

# Model_1 evaluation with perplexity metric #
Train_Prep = LDA.perplexity(Train_Data)
print("Train perplexity: ", Train_Prep)

Test_Prep = LDA.perplexity(Test_Data)
print("Test perplexity: ", Test_Prep)
print("_________________________________________________")

##### Model_2 Training #####
G_LDA = LdaModel(corpus=Corpus, id2word=Dict, num_topics=10)

print("Model_2 Training done.")

print("Topics found via Gensim LDA:")
# topic & keywords & weight(importance)
pprint(G_LDA.print_topics())
print("_________________________________________________")

# Model_2 evaluation with Coherence Score #
cm = CoherenceModel(model=G_LDA, corpus=Corpus, coherence='u_mass')
coherence = cm.get_coherence()
print("Coherence Score for =", coherence)


def calc_coherence_values(dictionary, corpus, end, start, step):
    coherence_values = []
    for num_topics in range(start, end, step):
        model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        coherence_model = CoherenceModel(model=model, corpus=corpus, coherence='u_mass')
        coherence_values.append(coherence_model.get_coherence())
    return coherence_values


coherence_list = calc_coherence_values(dictionary=Dict, corpus=Corpus, start=2, end=40, step=6)
End = 40
Start = 2
Step = 6
x = range(Start, End, Step)
plt.plot(x, coherence_list)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.show()
print("_________________________________________________")
