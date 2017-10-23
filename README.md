# marketMakerImitate
https://github.com/graham83/ML-examples.git

Utilising scikit learn MLPClassifer to try to imitate a basic market making algorithm in the Bund.  This is a very basic example of how supervised learning can be used to imitate a trading algorithm behavior, given the algorithms current position and relevant market data.  
This is a shallow neural network so a bit of expert input for feature engineering helps a lot e.g. Instead of looking at the orders in the context of price action, the features are derived from the time and sales order flow.  The network inputs represent order flow of time and sales in the market, broken down into buy and sell aggressor order size categories captured and the current market maker strategy position (flat, long, short).

Note: I have traditionally utilized AForge.NET and Encog in C# for machine learning, but these libraries are outdated and don't allow one to use the latest ML innovations e.g. AForge only has sigmoid activation which is much less efficient for training deep neural networks (as opposed to ReLu activation for example). Thus still learning the python ropes, and I sometimes find it quicker to write helper functions then find the relevant library (normalize in this example).

### Files
* inputs.csv contains the input features
* outputs.csv contains the associated output features
* mmPredictor.py  is the classifer

### Process
1. Read training data into X,Y array
2. Split data into training and test sets
3. Setup hyperparameters in the MLPCLassifier object
4. Run feed forward and backgward propagation on the training set using MLP.fit
5. Score the model on the test set using MLP.score

### ML 101 - Normalize your features
Using raw inputs, the learning curve is quite erratic and generalises poorly since the trainer stops after 2 consecutive training loss increases. A good reminder why it is standard practice to normalize your features.

![Market Maker No Normalization](https://github.com/graham83/marketMakerImitate/blob/master/Without%20Normalization.png)

Normalizing is performed by subtracting the mean and dividing by the std deviation. We get a smooth learning curve that gives nearly a 100% score. ie The neural network has learned to perform the exact same function as the market making algorithm. A perfect score is not unsurprising in hindsight as the algorithm is simply a number of basic logic operations (IF THEN BUY/SELL) which a neural network can learn. The 100% score also probably has to do with the training data being generated by another algorithm so there are no false positives/negatives in the labeling i.e The algorithm will always buy / sell when the same conditions are present.

![Market Maker No Normalization](https://github.com/graham83/marketMakerImitate/blob/master/With%20Normalization.png)

TO DO: 
1. Experiment with different activation functions to improve training performance (fewer iterations)
2. Experiment with different network sizes (how simple can this network be?)
3. Observe how regularization effects the training


