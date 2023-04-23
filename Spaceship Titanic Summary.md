Richard Hauser & Melchor Ronquillo

12/6/2022

Pre Processing

Python Links : https://colab.research.google.com/drive/1_dPTdoQQsEoTyW0ECD0wKN57lZsrR6Ix?usp=sharing

Grid search

https://colab.research.google.com/drive/1c8iFyi2kAZ8-7XDTehU3gLG1ZGKPPGeD?usp=sharing

Similar to the infamous “Titanic” dataset that has been used for statistical and machine learning applications, we are given the task to determine which passengers will be transported from the Space Titanic after it had “collided with a spacetime anomaly hidden within a dust cloud”. In this dataset, we are given many different variables such as: PassengerId - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is traveling with and pp is their number within the group. HomePlanet - The planet the passenger departed from. CryoSleep - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins. Cabin - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard. Destination - The planet the passenger will be debarking to. Age - The age of the passenger. VIP - Whether the passenger has paid for special VIP service during the voyage. RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the Spaceship Titanic’s many luxury amenities. Name - The first and last names of the passenger. Transported - Whether the passenger was transported to another dimension.

These variables are used in prediction when trying to determine whether someone was “transported” to a different dimension or not.

We first started out by preprocessing our data. Almost every column in this dataset had rows with NaN values, so we decided to establish that issue first. We analyzed the continuous variables “Room Service”, “FoodCourt”, “ShoppingMall”, “Spa”, “VRDeck” and determined that their distributions were not normal. Due to this, we decided to impute the NaN values in each of these columns with their respective median values. “Age” is another continuous variable, but since the data appeared to have a more normal distribution, we imputed the NaN values of this variable with the mean.

For each of the categorical variables such as “Home Planet”, “CryoSleep”, “Destination”, and “VIP”, we counted the frequency of every unique value within the column and imputed the NaN values with the value of highest instances (Mode) respectively. These variables were then converted into categorical data types, and used numbers to represent each specific unique value.

We dropped the Name Column as well with the cabin column as it was found that it didn’t have much significance to the model.

The methods that were used to build models were, Support Vector Machines(Our best Model), Gradient Boosted Models, Logistic Regression(in Python), and Random Forest.

Cross Validation was used to determine which was the best model.

We sliced the training data set into two sets. One with 5000 Observations, the other one with 3693 Observations. We trained on the 5000 and tested on the 3693.

Once we tuned all of the models, the SVM was found to have the lowest CV error with 19.57%

We then took that model and ran it through the testing dataset from kaggle which gave an accuracy of 79.237% Accuracy.

GBM WHEN TUNED HAS 20.44% ERROR. RANDOM FOREST HAS 19.66% ERROR SVM HAS 19.57% ERROR LOGISTIC REGRESSION 22.00% ERROR

In the future, we plan to continue to tune our models to achieve an accuracy of at least 80%. We would also want to go and try and fit a neural network on this data set and see how it would work and see if it would yield a larger classification rate. We could also take these classification models and try and apply them to different datasets too.
