Recurrent-Neural-Net
====================

## **Recurrent Neural Networks on International Airline Passenger's data**

#### We will be using 3 different type of Recurrent Network's :<br>

  - Simple Recurrent Neural Networks
  - Gated Recurrent Neural Networks
  - Long Short Term Memory Networks

**Why Recurrent Networks ?**

The network structure of Neural network is working fine, then why do we need Recurrent neural networks?
Answer to this question is variable size input data. For a neural network we need to build a Weight Metraix with the help of input data 
but as we do not have a set input size, its very hard to create a weight matrix for each dataset. 
Recurrent neural network models works on veriable input data. They are capable of taking indefinite length of input data for each dataset.
There are three main type of Recurrnet network mentioned above, LSTM is on of the most used model nowadays. 

There are gradient vanishing probelem in Recurrent Nural network models. But we are not going to discuss about it here, you can find so much
theory online to read about LSTM, GRU and RNN networks. Let's focus on the implementation part here. 

**Dataset:**

  - Dataset is taken from <a href="https://datamarket.com/data/set/22u3/international-airline-passengers-monthlytotals-in-thousands-jan-49-dec-60">datamarket.com</a>.<br>
  - Raw data source is available on <a href="https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv">Github</a>.<br>

**Workflow:**
  - Import libraries
  - Load dataset
  - Create a Neural network Model with Simple Recurrent Neural Net (RNN)
    - Change the window size (1,3,15) and record the accuracy on Train and Test.
  - Create a Neural network Model with Long Short Term Memory net (LSTM)
    - Change the window size (1,3,15) and record the accuracy on Train and Test.
  - Create a Neural network Model with Gated Recurrent Neural Net (GRU)
    - Change the window size (1,3,15) and record the accuracy on Train and Test.
  - Plot accuracy plot for all the model for comparison.
  
**Accuracy:**
Root Mean Square Error for the different model's on best Window size is given below:

- Train set 
1. Simple Recurrent Neural Networks - 10.28%
2. Gated Recurrent Neural Networks - 9.42%
3. Long Short Term Memory Networks - 11.26%

- Test set 
1. Simple Recurrent Neural Networks - 20.52%
2. Gated Recurrent Neural Networks - 18.72%
3. Long Short Term Memory Networks - 20.91%


*Files.docx file is provided with all file information and how to go through the code. Also result and record_observaitons files are provided with 
all the records and observations. Please go through those to get a clear understanding of how the Window size relate with the Error on Model.*

*Code snippets are included in record_observations file to show how code looks like for each models. Along with the Root mean square error plot for each
observation.*



