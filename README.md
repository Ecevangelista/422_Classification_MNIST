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


**K-Means Model**  

The MNIST data from google tensorflow data sets was downloaded to access the image data and the full list of explanatory variables was used to run the model.Mini-batch K-Means was run for the speediness of the algorithm.   

The initial model was run with n_clusters = 10 since there are 10 classes to predict, however, several more cluster sizes were used to assess the optimal number of clusters. From reviewing previous work on this dataset, optimal cluster size is 256, so I also tested sizes above and below this value: 200, 250, 256, 260, 265. 

I also ran silhouette diagrams on 10-15 clusters to view the distribution of the clusters.  

Average Silhouette Scores  
* 10 clusters: 0.059
* 11 clusters: 0.060
* 12 clusters: 0.061
* 13 clusters: 0.060
* 14 clusters: 0.056
* 15 clusters: 0.046

Silhouette Diagrams on Cluster Sizes 10-15  
Based on the silhouette diagram, n_clusters = 15 has the most evenly distributed cluster sizes, however, based on the average silhouette score, 12 clusters produced the highest average silhouette score with 0.061.  

![kmeans_sil10](https://user-images.githubusercontent.com/49419673/153773725-7f986656-a8da-42d7-a908-2cbce70cd37e.png)  

![kmeans_sil11](https://user-images.githubusercontent.com/49419673/153773732-a13eefce-b56e-4c4e-8e78-3f0a5382fe44.png)  
 
![kmeans_sil12](https://user-images.githubusercontent.com/49419673/153773794-dada8ab5-4364-40c9-8bc3-46e02348090f.png)  

![kmeans_sil13](https://user-images.githubusercontent.com/49419673/153773798-9086982c-3b82-448f-9170-88d9a0336e5e.png)  

![kmeans_sil14](https://user-images.githubusercontent.com/49419673/153773806-b3beca6f-4296-407b-9659-45ed1922d592.png)  

![kmeans_sil15](https://user-images.githubusercontent.com/49419673/153773814-a7e2797b-53f4-4aad-aa27-d4384a92c586.png)  


Since the labels are available to train the model, inertia, homogeneity, completeness, and accuracy are better metrics to asses the optimal cluster size. Lower inertia values represent the mean squared distance between each instance and its closest centroid. Homogeneity represents the criteria that each cluster contains only members of a single class, while Completeness is the criteria that all members of a given class are assigned to the same cluster.  

Based on balancing the 4 criteria, the cluster size of 256 achieves the highest homogeneity, completeness, and accuracy scores on the training dataset. While inertia decreases with larger cluster sizes, the other 3 criteria show decreased scores, so it’s not worthwhile to test higher cluster sizes.  

**K Means Cluster Performance**  
| Cluster Size | Inertia   | Homogeneity | Completeness | Accuracy |
| ------------ | --------- | ----------- | ------------ | -------- |
| 200          | 1,551,624 | 0.834       | 0.789        | 0.890    |
| 250          | 1,510,020 | 0.841       | 0.795        | 0.897    |
| 256          | 1,506,010 | 0.847       | 0.804        | 0.901    |
| 260          | 1,497,717 | 0.846       | 0.807        | 0.901    |
| 265          | 1,496,737 | 0.843       | 0.792        | 0.896    |  

**K Means Model Classification on 10 Clusters and 256 Clusters**

Increasing the cluster size to 256 improved F1 scores from 0.55 with 10 clusters to 0.90 on 256 clusters on the validation test dataset.  The model performed best on classifying "0", "1", and "5" with F1 scores of 0.96, and performed worst on classifying "9" with an F1 score of 0.83.  

Runtime on n_clusters = 256: 13 seconds

| K-Means model with 256 clusters   | F1 Score | Precision | Recall | Accuracy |
| --------------------------------- | -------- | --------- | ------ |--------- |
| **Total Model Weighted Average**  | 0.90     | 90%       | 90%    | 90%      |
|                                   |          |           |        |          |
| **Individual Digit Classes**      |          |           |        |          |
| 0                                 | 0.96     | 95%       | 97%    |          |
| 1                                 | 0.96     | 94%       | 99%    |          |       
| 2                                 | 0.93     | 94%       | 92%    |          |
| 3                                 | 0.88     | 87%       | 90%    |          |
| 4                                 | 0.85     | 91%       | 79%    |          |       
| 5                                 | 0.96     | 87%       | 85%    |          |
| 6                                 | 0.95     | 95%       | 96%    |          |
| 7                                 | 0.88     | 87%       | 89%    |          |       
| 8                                 | 0.87     | 91%       | 84%    |          |
| 9                                 | 0.83     | 79%       | 87%    |          |


### Conclusion

Since the K-Means dataset may have differed from the other 2 datasets, true comparison of the F1 scores is not available. The Random Forest Classifer achieved the highest performance with an F1 score of 0.97 and the range across all digits was consistent, achieving 0.95 or higher. All models behaved similarly in classification prediction, with the best performance seen with classifying "0" and "1", and the worst performance with classifying "8" and "9."  

The K-Means clustering model achieved the lowest runtime at 13 seconds. It also produced the widest range of F1 scores, ranging from 0.96 - 0.83, indicating that the model was not as successful at classifying some of the more difficult digits like "8" and "9." 




### References  
* sci-kit learn discussion on clustering performance evaluation  
https://scikit-learn.org/stable/modules/clustering.html#k-means

* Chanseok Kang's colab notebook for K-Means code on image classification
https://colab.research.google.com/github/goodboychan/goodboychan.github.io/blob/main/_notebooks/2020-10-26-01-K-Means-Clustering-for-Imagery-Analysis.ipynb



