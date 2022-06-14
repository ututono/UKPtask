## Assignment

Your task is to implement a simple sequence tagger using word embeddings for named entity recognition. In the second part of the task you
should suggest improvements to this simple setup.

#### Architecture & Training
Implement a sequence tagger using provided word embeddings for named entity recognition. The architecture must be a bidirectional LSTM with
a single 100-dimensional hidden layer. Use the following parameters for training:
* Use Crossentropy-loss and the Adam optimizer
* Train for 20 Epochs
* Set the batch-size to 1

#### Data
The data is already split into a train, dev, and test set. The input tokens are specified in the first column and the labels are in the last column.

The word embeddings are pretrained and should not be updated with the model. You can download them here:
https://nextcloud.ukp.informatik.tu-darmstadt.de/index.php/s/g6xfciqqQbw99Xw

#### Evaluation
1) Report the macro-averaged AND micro-averaged F1-scores on the dev data (for all 20 epochs) and of the final model on the test data. Do you observe
a difference between these two metrics? Please explain why or why not they result in similar performance and which you would prefer for the task.

2) Where does the model fail? Analyze and provide evidence for the errors your model makes in at most 8 sentences (and as many figures or samples you need).

3) What might cause the errors of your model? Provide suggestions for an improved architecture or alternative approaches. You may reason based on your
findings or any references to the literature.

Generally, feel free to add visualizations and data excerpts, where they support the evaluation and your line of argument.

#### Further requirements
Further requirements are:
* Use PyTorch (not Keras or Tensorflow) to implement the model
* Use python 3.6 or higher
* Do not use any other libraries besides native python libraries, PyTorch, numpy, pandas, or matplotlib (if you want to provide any visualization). 
* The resulting submission should be one (or several) Python files. Do not submit a Jupyter notebook. Please provide the files in a .zip.
