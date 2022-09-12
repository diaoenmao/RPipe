# RPipe
This is a Pipeline for machine learning Research (RPipe)
 
## Requirements
See `requirements.txt`

## Instruction
- Use make.sh to generate run script
- Use make.py to generate exp script
- Use process.py to process exp results
- Hyperparameters can be found in config.yml and process_control() in utils.py

## Examples
 - Generate run script
    ```ruby
    bash make.sh
    ```
 - Generate run script
    ```ruby
    python make.py --mode base
    ```
 - Train with MNIST and linear model
    ```ruby
    python train_classifer.py --control_name MNIST_linear
    ```
 - Test with CIFAR10 and resnet18 model
    ```ruby
    python test_classifer.py --control_name CIFAR10_resnet18
    ```
 - Process exp results
    ```ruby
    python process.py
    ```

## Results
- Learning curves of MNIST
![MNIST_Accuracy_mean](/asset/MNIST_Accuracy_mean.pdf)

- Learning curves of CIFAR10
![CIFAR10_Accuracy_mean](/asset/CIFAR10_Accuracy_mean.pdf)


## Acknowledgement
*Enmao Diao*
