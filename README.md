# Redditcontentanalysis
Since the advent of social media, data transmission has improved tremendously. People could share data(text, video, images, audio, etc;) in the form of posts. The most common platform where people post about diverse topics is Reddit. From the Reddit Submissions dataset, we have tried to predict the success of the post or submission, based on different features such as the submission time, posted community, title length, etc; We have used several Regression, TF-IDF, transformer models to evaluate the co-dependencies among different features in the Dataset.

<br>• In this project, we have built a sentence transformer model for classifying the Reddit titles with the subreddit classes, we have used all-mpnet-base-v2 and multi-qa-mpnet-base-dot-v1 semantic search models to map reddit titles with 473-dimensional dense subreddit vector spaces.
<br>• We have also built word2vec, unigram, bigram and similarities models from the most correlated variables which we got from exploratory data anaysis
<br>• Further, we have evaluated the models using accuracy and precision, recall metrics. Accuracy was found to be 60% for sentence transformer models
