# Digit Recognizer
<pre>

________  .__       .__  __    __________                                  .__                     
\______ \ |__| ____ |__|/  |_  \______   \ ____   ____  ____   ____   ____ |__|_______ ___________ 
 |    |  \|  |/ ___\|  \   __\  |       _// __ \_/ ___\/  _ \ / ___\ /    \|  \___   // __ \_  __ \
 |    `   \  / /_/  >  ||  |    |    |   \  ___/\  \__(  <_> ) /_/  >   |  \  |/    /\  ___/|  | \/
/_______  /__\___  /|__||__|    |____|_  /\___  >\___  >____/\___  /|___|  /__/_____ \\___  >__|   
        \/  /_____/                    \/     \/     \/     /_____/      \/         \/    \/       
</pre>
## ABOUT
Simplest Machine Learning Model to recognizing hand written digits.

### MNIST Dataset

- Run the following commands to download datasets
```sh
wget https://myawsbucket003.s3.ap-south-1.amazonaws.com/AI+ML/Digit+Recognition/datasets/mnist_test.csv\n

wget https://myawsbucket003.s3.ap-south-1.amazonaws.com/AI+ML/Digit+Recognition/datasets/mnist_train.csv
```


MNIST consists of 60,000 handwritten digit images of all numbers from zero to nine. Each image has its corresponding label number representing the number in image.<br>

Each image contains a single grayscale digit drawn by hand. And each image is a 784 dimensional vector (28 pixels for both height and width) of floating-point numbers where each value represents a pixel’s brightness.

the data sets also provided in notebook.

> #### Note:<br>
> After downloading update path `train_data_file`, `test_data_file` in `__init__` method

## KNN

`K Nearest Neighbors` is a classification algorithm. It classifies the new data point (test input) into some category.
To do so it basically looks at the new datapoint’s distance from all other data points in training set.
Then out of the k closest training datapoints the class in majority is assigned to that new test data point.<br>
*Pretty Simple Right* :)

### _Finding Distance_

The `LN Norm distances` gives the distance between two points.

 > If n is 1, the LN Norm is Manhattan distance<br>
 > If n is 2, it is Euclidean distance

By using LN Norm, we can find distances between test input and training input

>![distance](https://latex.codecogs.com/png.image?\dpi{110}%20\sqrt[n]{\sum_{i=1}^{n}\left|a_{i}-b_{i}&space;\right|^{n}})


### _Get K Nearest Neighbours_

After finding the distances between test and train inputs, to know in which classification the test input would be, we need to get k nearest neighbours.

### _Voting_

> #### _Distance Voting_
>Among KNN there could be voting tie,<br>
>One of the way to broke voting tie is Distance Weighted KNN, i.e., `sum of inverse of distances of all predicted labels`<br><br>
> ![equation](https://latex.codecogs.com/svg.image?\bg_white&space;\sum&space;\frac{1}{distance})
> <br>for all k nearest neighbours in KNN

> #### _Majority Voting_
>Among KNN there could be voting tie,<br>
>Another of the way to broke voting tie is Majority Based KNN, i.e., `out of predicted labels which label is most repeated that label does chosed`

### _Hyperparameter Tuning_

Choosing what are the best values of  `n` and `k` that gives the best accuracy.
