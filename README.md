#                           MNIST Digits Regression EC with K-fold validation

We used MNIST digits dataset to calculate the regression ECs. 

## Model Description

1) Two fully convolutional layers, 
2) Relu activation function and MaxPooling,
3) Mean Squared Error (MSELoss) as loss function, 
4) Stochastic Gradient Descent (SGD) as optimizer,
5) Learning Rate 0.01
6) Number of Epochs 50
7) K-fold (k=5) external holdout cross-validation


## Regression ECs and Goodness of fit for the MNIST Digits using CNN

Model | EC Method: Value | MAE | MAPE | MSqE | R2
---------- | ---------- | ---------- | ---------- | ---------- | ---------- | 
CNN | Ratio (0.10) <br /> Ratio-diff (0.40) <br /> Ratio-signed (0.05) <br /> Ratio-signed-diff (0.09) | 0.56 (0.06) | 0.22 (0.32) | 0.67 (0.10) | 0.92 (0.01)


## Model Prediction VS Actual Number


Prediction| 7.0|  2.0| 1.0| 0.0| 3.9| 0.8| 4.3| 6.3| 6.3| 8.9 
-----------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
Actual| 7| 2| 1| 0| 4| 1| 4| 9| 5| 9|


## Model Train using K-fold and Collect Regression EC (Test)

```
python3 run.py --choices=Test
```

<!-- ## Model Test

```
python3 run.py --choices=Test
``` -->

## Model Compare (with actual vs prediction)

```
python3 run.py --choices=Compare
```

## References

The model is inherited from this [link] on september 15 2021. However, we changed the loss function, optimizer and the last layes has only one output neuron.

[link]: https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118
