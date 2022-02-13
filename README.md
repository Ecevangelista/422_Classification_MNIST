# Classification Using PCA and K-Means on MNIST Digit Recognizer Dataset  

The exercise will look at the performance of PCA and the K-Means clustering algorithm on the MNIST Digit Recognizer Dataset. The classification models will produce predictions of digits 0-9 on handwritten images of these numbers. A Random Forest model was built as a baseline. The metrics used to assess performance will include the F1 score, Precision, Recall and Accuracy.  

### EDA and Preprocessing  
The train dataset from Kaggle consists of 785 columns and 42,000 rows. Columns include “label,” representing the number that is depicted in the images, and the remaining columns consist of pixel positions. The rows contain a pixel value ranging from 0 to 255. “Label” is the response variable used to train the dataset.

A review of the distribution of label values 0-9 revealed that the classes are balanced.  
![label countplot](https://user-images.githubusercontent.com/49419673/153771969-ae3904ad-4f6e-497a-9ae3-90136bfd8825.png)

Preprocessing consisted of dropping columns from Pixel0 to 32 and Pixel779 to 783 to eliminate columns with only “0s.” This reduced the explanatory variables from 785 to 745.   

For PCA and the KMeans model, the explanatory variable data had to be scaled using StandardScaler or numpy to transform the data into values between 0 and 1.  

Prior to building the models, the train dataset for the Random Forest and PCA with Random Forest models were split into train and test groups with an 80%/20% split to be able to validate the models. The K Means model was split using a 85%/15% split due to accessing the data from google tensorflow.  

### Model Evaluation  
**Random Forest**  

The Random Forest model was run with default hyperparameters (n_estimators = 100) and produced an overall weighted average F1 score of 0.97. The model performed best on predicting “0”s, “1”s, and “6”s, with F1 scores of 0.98 and worst on predicting “8” and “9” with F1 scores of 0.95.   

Runtime: 24 seconds

| Random Forest                     | F1 Score | Precision | Recall | Accuracy |
| --------------------------------- | -------- | --------- | ------ |--------- |
| **Total Model Weighted Average**  | 0.97     | 97%       | 97%    | 97%      |
|                                   |          |           |        |          |
| **Individual Digit Classes**      |          |           |        |          |
| 0                                 | 0.98     | 98%       | 98%    |          |
| 1                                 | 0.98     | 98%       | 99%    |          |       
| 2                                 | 0.96     | 96%       | 96%    |          |
| 3                                 | 0.96     | 96%       | 96%    |          |
| 4                                 | 0.97     | 96%       | 98%    |          |       
| 5                                 | 0.97     | 97%       | 97%    |          |
| 6                                 | 0.98     | 97%       | 98%    |          |
| 7                                 | 0.97     | 97%       | 96%    |          |       
| 8                                 | 0.95     | 96%       | 93%    |          |
| 9                                 | 0.95     | 96%       | 95%    |          |

**PCA with Random Forest**

Principal Components Analysis was conducted to reduce the dimensions of the dataset before running the Random Forest model. To preserve 95% of the variance in the data, the dataset was reduced to 315 components from 745.  The Random Forest model was run with default hyperparameters (n_estimators = 100) and produced an overall weighted average F1 score of 0.93.  

The model performed best on predicting “0” and “1”, with F1 scores of 0.98 and 0.96 and worst on predicting “8” with F1 score of 0.89 on the validation test dataset.  

Runtime to fit PCA: 6 seconds  

Runtime for PCA and Random Forest: 1 minute, 16 seconds  

| PCA & Random Forest               | F1 Score | Precision | Recall | Accuracy |
| --------------------------------- | -------- | --------- | ------ |--------- |
| **Total Model Weighted Average**  | 0.93     | 93%       | 93%    | 93%      |
|                                   |          |           |        |          |
| **Individual Digit Classes**      |          |           |        |          |
| 0                                 | 0.96     | 96%       | 97%    |          |
| 1                                 | 0.98     | 98%       | 98%    |          |       
| 2                                 | 0.91     | 90%       | 92%    |          |
| 3                                 | 0.91     | 88%       | 93%    |          |
| 4                                 | 0.94     | 94%       | 95%    |          |       
| 5                                 | 0.90     | 92%       | 88%    |          |
| 6                                 | 0.95     | 94%       | 96%    |          |
| 7                                 | 0.93     | 93%       | 94%    |          |       
| 8                                 | 0.89     | 93%       | 86%    |          |
| 9                                 | 0.90     | 92%       | 89%    |          |



