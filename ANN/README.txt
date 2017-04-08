##################
#   Alex Marion  #
#     CS383      #
#  Homework 8    #
##################

===== README =====

Files included in this submission are listed below:
    1. README
    2. HW8.pdf
    3. HW8_problem1and2.m
    4. HW8_problem3.m
    5. binary_threshold_classifier.m
    6. read_spam_data.m
    7. read_CTG_data.m
    8. standardize_data.m
    9. spambase.data
    10. CTG.csv

Coding Assignment:
    1. Artificial Neural Networks: The four .m files (HW8_problem1and2.m, binary_threshold_classifier.m, read_spam_data.m, and standardized_data.m) are required to run the first coding portion of the assignment. The spambase.data file is also required and these files must be in the same directory. Run HW8_problem1and2.m to train an artificial neural network on the spambase data and output the Accuracy and plot the training error. The output of the script is included in HW8.pdf.

    2. The Precision-Recall Tradeoff: The same files required for part 1 are required for part 2. Running the HW8_problem1and2.m file will also plot the precision recall for the artificial neural network trained on the spambase data varying the threshold for binary classification between 0 and 1 by 0.1. 

    3. Multi-Class Artificial Neural Networks: The three .m files (HW8_problem3.m, read_CTG_data.m, and standardized_data.m) are required to run the third coding portion of the assignment. The CTG.csv file is also required and these files must be in the same directory. Run HW8_problem3.m to train an artificial neural network on the CTG data and output the Accuracy and plot the training error. The output of the script is included in HW8.pdf.

TO RUN: Either load the code in matlab (and include the folder in your path) and run HW8_problem1and2.m and HW8_problem3.m
	OR
	Run the following commands on tux:
      matlab -r HW8_problem1and2
      matlab -r HW8_problem3
	This will print the output of HW8_problem1and2.m and HW8_problem3 respectively. 
