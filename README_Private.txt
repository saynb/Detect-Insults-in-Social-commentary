The two lists in project_preproess files are

1.	The list with 0 and 1 values for the corresponding comments.

2.	The list of list of all words.

When you'll run the file, they both will get printed, see the results.

Check if they are fitting without any errors in countVectorizer and it is working.

TO DO:

1.	Since all the stopwords have been removed, we have to allow some words to be there (as discussed).
	We'll see this later. First we need a basic working model (maybe with low precision but atleast working).

2	There is a very high probability that if a bad word/insulting word is used, then the probability of a 
	comment being an insult is high/ So, along with the tfidf feature, we are also using a list of bad words 
	and calculating the bad word frequency in a particular comment to arrive at pur final results.





Visualizations:

1.	Combined Learning Curve of all the models which we are trying.
2.	Histogram of all f1 score of the models 
3.	Scatter plot and learning curve of the model which we will be using for the best configuration.

Things we gonna focus : 
Importance of our project
How it will h help the social media and public forums to automatically detect insults and obligatory comments
(1-2 mins)

1.	Custom stop words list (We need to speak atleast 2-3 minutes about the preprocessing of data)
2.	Bad words transformer combines with tfidf transformer (Around 4-5 minutes on our classifier models and the results)
3.	K means clustering applied (How we are making clusters and what was the result)
4.	Our best results along with the visualization of our results.