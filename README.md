# Wikipages-Sorting
Class Project for web scraping Wikipedia articles to cluster and classify different genres


Wikipedia page urls are listed in script, using beautiful soup to parse the html paragraphs into a dataframe as plain text.

Our dataframe is organized into a frequency table and Latent Semantic Analysis is applied. 

Multiple methods in script for distance measures, differing ngram word count, and different degrees of natural language processing on the vectorized articles.

Using just frequency distribution in training model for K-nearest neighbors on articles, shows the mean accuracy for 30 trials. 
Then implements term frequency-inverse document frequency method to weight terms that appear more frequently throughout multiple articles. 
Logistic and Random Forest Regression is applied. 

Finally use Singular Value Decomposition (SVD) on frequency distribution dataframe, considered Latent Semantic Analysis (LSA), in which we will project feature vectors to a smaller subset, with the number of words in an article near the 10,000's, we can remove "stop words" (ex: "the", "a", "if", etc.) and project onto the most prominent 1000 features for each article to better catagorize & classify.


After comparing different methods of classification attempt to cluster with a k-means algorithm on same LSA vectorized articles. Creating a Scree plot for different assumed number of clusester with corresponding statistical measures for each iteration.

There are limitations in this way of comparing natural language processing methods in that the catagorization of the articles depends on the usefulness of the catagorization, and a criteria that people may not consider or is hard to notice, might be better for sorting say articles between positive and negative insinuations or some other metric. Research and additional work needed to push this project into semantic analysis and clustering/classification of news articles. 
