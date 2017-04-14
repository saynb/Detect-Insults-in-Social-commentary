CSE 537 Project - Detection of Insults in Social Commentary

Team Details:
	Members	- 	Hemant Pandey		: 110828730
				Sayan Bandyopadhyay	: 110946522
				Snigdha Kamal		: 110937472
				 

Main Project Folders/File Descriptions:
	- ./
		Contains entire codebase related to project processing, datasets, python files along with other resources

			* Project_preprocess_data.py : File containing which cleans, preprocesses the data. For detailed information, see the project report.

			* Train_data.py : Contains the entire training and modelling code.

			* Plot.ipynb : Contains the code for the visualisation of graphs and charts.

			* train : Training data

			* test : Test data

			* train_clean : Preprocessed data from Project_preprocess_data.py

			* test_clean : Preprocessing the test data.

			* Contributors : Members involved in the project

			* full_list_of_bad_words : Google bad words list used.

			* README : The file that you are reading right now

			* .png and .jpg files : Graphs and plots generated dynamically

Pre requisites:
	1.) Install Jupiter ipython for the functioning of plot.ipynb (This includes graphs for understanding of data)

Steps to Run:
	
	1.)	Run train_data.py to see the results and the learning curves (This is the main code which trains the data)
		It will take some time to train all the data based on five models, print the results.
		At the end, a pkl will will be generated on basis of the best configuration (which is a part of submission)
		and can be used as a model for future classifications.

You can clone the git repo at : https://github.com/saynb/Detect-Insults-in-Social-commentary.git
Code Repository : https://github.com/saynb/Detect-Insults-in-Social-commentary

