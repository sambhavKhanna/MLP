I have used tanh as the activation function. The loss function is represented along with the network in the images below. ```[[1.0, -2.0], 
      [3.0, 4.0]]``` are the two inputs passed to the network, and ```[0.9, -0.04]``` are the expected outputs for the corresponding inputs.
This is the starting network with 2 nodes in the input layer and 1 node in the output layer:![image](https://github.com/sambhavKhanna/MLP/assets/125531539/ccad4b13-caf5-4e05-b426-18d6bb21b613)
This is the network after one backward pass:![image](https://github.com/sambhavKhanna/MLP/assets/125531539/a5b0c270-49de-4ae8-9736-76f5e8d8df07)The gradients have been updated.
The final network after 20 iterations of gradient descent is:![image](https://github.com/sambhavKhanna/MLP/assets/125531539/71ccd338-59da-49e6-903d-421d74cf835b)
The initial prediction was ```[-0.0986, -0.8374]``` and after 20 iterations it is ```[0.8604858134000907, -0.48475656988391974]``` with a loss of ```0.14886848744469197```. These are the values for the loss function at each iteration:
```
0 Value(data=1.4523656443334825)
1 Value(data=0.6593573706609923)
2 Value(data=0.17041638445409618)
3 Value(data=0.08834727942232476)
4 Value(data=0.056515880669422466)
5 Value(data=0.04058855522348429)
6 Value(data=0.03207239611752707)
7 Value(data=0.02899800661323352)
8 Value(data=0.03308453547415589)
9 Value(data=0.048395576665073894)
10 Value(data=0.087070904004647)
11 Value(data=0.130200860599883)
12 Value(data=0.1925455596364463)
13 Value(data=0.15536506825222096)
14 Value(data=0.20595418399590965)
15 Value(data=0.14933858522874016)
16 Value(data=0.2015656492202712)
17 Value(data=0.14950530401353004)
18 Value(data=0.20054203979021923)
19 Value(data=0.14886848744469197)
```    
