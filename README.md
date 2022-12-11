# Topic-Modeling
Topic modeling is an unsupervised machine learning technique that’s capable of scanning a set of documents, detecting word and phrase patterns within them, and automatically clustering word groups and similar expressions that best characterize a set of documents.
The objective is to identefy some topics throught frequently repeted words.

Methodology:
Using the Latent Dirichlet Allocation (LDA) algorithm.
Comparison between Gensim LDA model and Sklearn LDA model to compare the results.

Data summary:
The data that was used to train the models is articles dataset.
I used the content column to train the models. It’s a column of strings with 50000 record.

![data](https://user-images.githubusercontent.com/88488379/206722067-9a40a1d1-ed7e-46f6-8a59-0f65f86335df.PNG)
![data2](https://user-images.githubusercontent.com/88488379/206722106-3dc03407-8663-4fba-a816-30d3120328bf.PNG)

Most common words in the dataset using wordcloud.
![WordCloud](https://user-images.githubusercontent.com/88488379/206925293-c26fa5a7-2bf5-4903-908f-f55accb9158c.PNG)

Results:
Sklearn model results:
![image](https://user-images.githubusercontent.com/88488379/206925358-f13a7bc9-ccad-451d-a886-851a9fa9258e.png)
![image](https://user-images.githubusercontent.com/88488379/206925367-77f7f0a3-3d5f-4a16-acb9-7d8b5fbf6e2e.png)
Gensim model results: 
![image](https://user-images.githubusercontent.com/88488379/206925394-daf04909-809b-449c-a0c6-6d83573d674f.png)
![image](https://user-images.githubusercontent.com/88488379/206925402-3d99d103-233b-4081-a9c3-09f0f350ec08.png)
