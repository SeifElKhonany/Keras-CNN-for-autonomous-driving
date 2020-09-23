# Keras-CNN-for-autonomous-driving
A Convolutional Neural Network, created for my autonomous vehicle graduation project, that maps input images to discrete output steering angles. This CNN resulted in 94% training accuracy with 91% testing accuracy. Integrated with path planning and sensor fusion using ROS2, it produced a functional autonomous vehicle as you can see in these [videos](https://l.facebook.com/l.php?u=https%3A%2F%2Fdrive.google.com%2Fdrive%2Ffolders%2F1OwAMyG7fHQiTadoCMDduKB3sZYNETCZo%3Ffbclid%3DIwAR0Ini7hzPhscoFH9otqEZh8Zn4RK1THo6sjPXS5Ou2MRsNlsi2yFvRb9ss&h=AT0vFdJWYaJIeatbpAScYBGUWrmAYYMjiitejweq-kd1bTUM31cUCurXcMl4dQZTtpPy7K050pHrZEANfnhcScfSNCbdoKtr5WO3xK1K5CaMi5EC5wVDdF4-ZqmmK97RDrAy).

## How to use
* Set up a python environment and install Tensorflow.
`pip install tensorflow-gpu`
* Assign values to these variables suitable to your project.
```python
bs = 10   # batch size
num_epochs = 2

n_training = "Number of training images"
n_validation = "Number of validation images"

train_path = r'path to training images folder'
valid_path = r'path to validation images folder'

input_shape = (376, 672, 3)
```
* Use the saved model to predict steering angles based on images.
