# Classification Using PCA and K-Means on MNIST Digit Recognizer Dataset  

The exercise will look at the performance of PCA and the K-Means clustering algorithm on the MNIST Digit Recognizer Dataset. The classification models will produce predictions of digits 0-9 on handwritten images of these numbers. A Random Forest model was built as a baseline. The metrics used to assess performance will include the F1 score, Precision, Recall and Accuracy.  

### EDA and Preprocessing  
The train dataset from Kaggle consists of 785 columns and 42,000 rows. Columns include “label,” representing the number that is depicted in the images, and the remaining columns consist of pixel positions. The rows contain a pixel value ranging from 0 to 255. “Label” is the response variable used to train the dataset.
