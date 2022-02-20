# Classification using Neural Networks with Keras on MNIST Digit Recognizer Dataset  

### Overview  
The exercise will look at the performance of several neural networks created with Keras on the MNIST Digit Recognizer Dataset. The classification models will produce predictions of digits 0-9 on handwritten images of these numbers.  

Neural networks will be tested in the following 2x2 crossed design:  
* 1 layer/100 nodes  
* 1 layer/300 nodes  
* 2 layers/100 nodes  
* 2 layers/300 nodes  

Of the models with 2 layers, the worst-performing one will be further developed using the dropout layers to improve performance.  

Performance metrics will include the F1 score, Precision, Recall, Loss, and Accuracy.  

### Model Performance  

Model 2 with 1 layer and 300 nodes achieved the highest performance with 98% testing accuracy and a 0.98 F1 score. The F1 score for classifiying all digits was 0.97 or higher, with “1” achieving an F1 score of 0.99. The model produced the lowest F1 scores when classifying “3”, “5”, and “8”, indicating that digits with rounded forms were most difiicult. Model 2’s confusion matrix further illustrates this difficulty when showing that these digits were incorrectly predicted amongst each other:  
* “3” had the highest occurrence of being incorrectly predicted as “5” and "8"  
* “5”  had the highest occurrence of being incorrectly predicted as “3”
* "8" had the highest occurence of being incorrectly predicted as "3"  

Model 4 was the worst-performing of all models due to the lowest individual digit F1 score produced at 0.96. Model 4 (2 layers/300 nodes) produced the 0.96 F1 score when classifying “8” and “9.” Similar to Model 2, “8” had the highest occurrence of being incorrectly predicted as “3”, while “9” had the highest occurrence of being incorrectly predicted as “7”.  

**Training Time, Loss, and Accuracy**  
| Model: Layers/Nodes    | Time          | Training Loss | Training Accuracy | Testing Accuracy | 
| ---------------------- | --------------| ------------- | ----------------- |----------------- |
| #1: 1 layer/100 nodes  | 1 min 14 secs | 9.782 e-04    | 100%              | 97.4%            |
| #2: 1 layer/300 nodes  | 2 min 22 secs | 2.713 e-05    | 100%              | 97.9%            |
| #3: 2 layers/100 nodes | 1 min 37 secs | 0.002         | 100%              | 97.3%            |
| #4: 2 layers/300 nodes | 2 min 22 secs | 0.001         | 99.7%             | 97.3%            |       


**F1 Score, Precision, Recall**
| Model: Layers/Nodes    | F1 Score | Precision | Recall |
| ---------------------- | -------- | --------- | ------ |
| #1: 1 layer/100 nodes  | 0.97     | 97%       | 97%    |
| #2: 1 layer/300 nodes  | 0.98     | 98%       | 98%    |
| #3: 2 layers/100 nodes | 0.97     | 97%       | 97%    |        
| #4: 2 layers/300 nodes | 0.97     | 97%       | 97%    |              


**Confusion Matrices and Classification Reports**  
Confusion matrices will have the True Class as the first row and the Predicted Class as the first column.  

Model 1  
![Conf matrix model 1](https://user-images.githubusercontent.com/49419673/154863282-b4aa4548-0131-48fc-a904-588463a32afa.png)
![classification report 100a](https://user-images.githubusercontent.com/49419673/154863290-7973640c-f306-4d85-bc17-38ff64749af5.png)

Model 2  
![Confusion matrix model 2](https://user-images.githubusercontent.com/49419673/154863319-f321f9e3-d5cf-4b19-89ee-8455240b1f72.png)
![classification report 300a](https://user-images.githubusercontent.com/49419673/154863331-123af994-3de3-462e-9ae3-475ceaae0ca6.png)

Model 3  
![Confusion matrix Model 3](https://user-images.githubusercontent.com/49419673/154863339-62cb6041-a346-4d0b-a94a-9cee3e6bdbb7.png)
![Classificaton report 100_2b](https://user-images.githubusercontent.com/49419673/154863351-f4577434-6a71-436f-a844-f887bea0ffdf.png)

Model 4  
![Confusion Matrix Model 4](https://user-images.githubusercontent.com/49419673/154863366-97b0b6ae-554f-4605-a503-ba4f305b2b44.png)
![classification report 3002 USE](https://user-images.githubusercontent.com/49419673/154863375-2d908b41-0362-4867-b908-29d1386bf58e.png)


### Improving performance with dropout layers
Since Model 4 had the worst performance, it was further developed as Model 5 with dropout layers,  with 20% of the features “dropped out.”  Model 5 surpassed both 2-layer models, achieving 97.7% testing accuracy and an F1 Score of 0.98. In Model 5, all digits achieved an F1 score of 0.97, or higher, whereas, in Model 4, the lowest F1 score was 0.96. Model 2 still performed slightly better than Model 5 when looking at individual digit F1 scores, as the lowest F1 scores for Model 2 were produced for only 3 digits, whereas Model 5 produced the lowest F1 scores for 4 digits.  

| Model: Layers/Nodes                 | Time          | Training Loss | Training Accuracy | Testing Accuracy | 
| ----------------------------------- | --------------| ------------- | ----------------- |----------------- |
| #5: 2 layers/300 nodes/20% dropout  | 3 min 22 secs | 0.003         | 99.9%%            | 97.7%            |

| Model: Layers/Nodes                 | F1 Score | Precision | Recall |
| ----------------------------------- | -------- | --------- | ------ |
| #5: 2 layers/300 nodes/20% dropout  | 0.98     | 98%       | 98%    |

Model 5 Confusion Matrix and Classification Report  
![Confusion Matrix Model 5](https://user-images.githubusercontent.com/49419673/154863465-e0a1ffa0-0de3-4126-b65b-5218e618c9d1.png)
![classification report drop model 3002b](https://user-images.githubusercontent.com/49419673/154863480-79d7f970-eb38-4374-8ffd-9c6c856290c3.png)


### Conclusion  
Model 2 with 1 layer/300 nodes achieved the highest testing accuracy (97.9%) and F1 score (0.98). Applying dropout layers to the worst-performing model (Model 4) boosted performance to closely match Model 2, by producing a testing accuracy of 97.7% and F1 score of 0.98.  
