IoT Device Classification Pipeline

This project implements a two-stage machine learning pipeline to classify IoT devices based on their network behavior. The pipeline processes feature files, applies Naive Bayes for categorical data (ports, domains, ciphers), and uses a custom Random Forest classifier to predict the device type.

Setup and Installation
1.
Prerequisites
Make sure you have Python 3.6 or newer installed. You will also need to install the following Python libraries:

- numpy
- scikit-learn
- tqdm
**To install all the dependencies, you can just pip install -r requirements.txt**

2.
Dataset
You need a properly formatted dataset containing feature files for various IoT devices. Ensure that each device type has its own directory containing the corresponding feature files in JSON format.
The feature files should contain the necessary data for network flow analysis. Make sure the dataset is organized and formatted correctly for the pipeline to work effectively.

3.
How to Run

1. Prepare your dataset by organizing the data into directories, with each device class (deviceA, device, etc) containing the relevant JSON feature files.
   
2. Run the pipeline by executing the following command:

python3 classify.py 

run the python file classify, and in root pass it the IOT data folder as its parameter, in this example we give the relative filepath from the folder containing classify.

Once the pipeline has run, it will display a classification report with metrics such as:
- Accuracy
- Precision
- Recall
- F1-score
