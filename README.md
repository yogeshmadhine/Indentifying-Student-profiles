# Indentifying-Student-profiles across online judges using explainable AI

Online Judge (OJ) systems automate grading of
programming assignments, making it fast and accurate.
OJ systems usually provide limited feedback, only
indicating if the code is correct.
Using Educational Data Mining (EDM), we analyze
submission data to predict student success and
behaviors.
Key data includes the number of submissions and
submission timing.
We use Multi-Instance Learning (MIL) and Explainable
Artificial Intelligence (XAI) to make predictions
understandable.
Tested on data from three years with 2,500 submissions,
our method accurately models student profiles and
predicts success.


![Screenshot 2024-08-04 133051](https://github.com/user-attachments/assets/815cc160-186d-4023-88a3-0e9f540df6d6)



![Screenshot 2024-08-04 133348](https://github.com/user-attachments/assets/a7a32871-b519-4a1b-96cb-7af0249c1a6e)

Registration Process
The registration process involves filling out a form with various fields to create an account on the platform. Here are the steps and details for each field:

Enter Username:

Input your desired username in the provided text box.
Example: "Yogesh"
Enter Email Id:

Enter a valid email address.
Example: "mail@gmail.com"
Enter Gender:

Select your gender from the dropdown menu.
Example: "Male"
Enter Country Name:

Type the name of your country.
Example: "India"
Enter City Name:

Type the name of your city.
Example: "City"
Enter Password:

Create a password for your account.
Example: (hidden for security)
Enter Address:

Provide your address details.
Example: "address"
Enter Mobile Number:

Enter your mobile number.
Example: "5656468"
Enter State Name:

Type the name of your state.
Example: "state"
Register:

After filling out all the fields, click the "REGISTER" button to submit your details and create an account.

![Screenshot 2024-08-04 133126](https://github.com/user-attachments/assets/88be19d4-7cb7-4f04-93f6-1e9efb862223)

![WhatsApp Image 2024-08-04 at 15 16 58_0f58d193](https://github.com/user-attachments/assets/8a98a1f6-3729-4de6-b286-2c4ce7b7ac4e)

Inputs
The form requires several inputs related to the student:

Fid (ID): A unique identifier for the student.
Gender: The gender of the student.
Parental Level of Education: The highest education level achieved by the student's parents.
Race/Ethnicity: The race or ethnicity group the student belongs to.
Math Score: The student's score in mathematics.
Writing Score: The student's score in writing.
Solving Tasks by Time: The number of tasks solved by the student within a given time.
Age: The age of the student.
Lunch: The type of lunch the student receives (standard or free/reduced).
Degree Type (degree_t): The type of degree the student is pursuing (e.g., Science & Technology).
Test Preparation Course: Whether the student has completed a test preparation course.
Reading Score: The student's score in reading.
Internships: The number of internships the student has completed.
Tasks Submitted on Date: The number of tasks submitted by the student on a specific date.
Prediction
Once these inputs are filled in, the system uses a trained dataset to predict the student's profile. This process involves:

Data Collection: The system collects data about students, including their scores, demographics, and other relevant information.
Training the Model: Using machine learning algorithms, the system is trained on this dataset to learn patterns and relationships between the inputs and the desired output (student profile).
Making Predictions: When new data is entered into the form, the system uses the trained model to predict the student's profile. This could include various aspects such as academic performance, likelihood of success in certain subjects, or other profile-related predictions.
Explainable Artificial Intelligence (XAI)
Explainable AI is crucial here as it allows the predictions made by the system to be understood by humans. This means the system can provide insights into why a particular prediction was made, based on the input features. This transparency helps in:

Building Trust: Users can understand and trust the predictions.
Identifying Bias: Ensuring that the model is fair and not biased towards any group.
Improving the Model: Providing feedback to improve the accuracy and fairness of the model


![WhatsApp Image 2024-08-04 at 15 22 07_7336d358](https://github.com/user-attachments/assets/2867a9bb-5ef5-4350-bb66-009bacc1ecc1)

The navigation bar (navbar) at the top of the page contains several options. Here's an explanation of each option in the navbar:

Navbar Options
Browse Students Datasets and Train & Test Data Sets

Explanation: This option allows users to browse through the datasets that contain information about students. It includes both the training datasets (used to train the machine learning models) and the testing datasets (used to evaluate the model's performance).
View Trained and Tested Accuracy in Bar Chart

Explanation: This option provides a visual representation of the accuracy of the trained models. The bar chart displays the accuracy metrics, helping users understand how well the models perform on the training and testing datasets.
View Trained and Tested Accuracy Results

Explanation: Similar to the bar chart option, this provides detailed accuracy results in a more comprehensive format. Users can view numerical accuracy metrics and possibly other evaluation metrics (e.g., precision, recall) for the trained models.
View Prediction Of Online Student's Profile Judgement

Explanation: This option allows users to see the predictions made by the system for students' profiles. It uses the input data provided to generate predictions about various aspects of the students' profiles.
View Online Student's Profile Judgement Ratio

Explanation: This option shows the ratio of different judgments made by the system. It provides an overview of how many students fall into different profile categories based on the predictions.
Download Predicted Data Sets

Explanation: This option enables users to download the datasets containing the predicted profiles of students. It is useful for further analysis or record-keeping.
View Online Student's Profile Judgement Type Ratio Results

Explanation: This option provides detailed ratio results of different types of judgments made by the system. It may include statistical summaries or visual representations of the distribution of profile types.
View All Remote Users

Explanation: This option displays a list of all remote users who have used the system. The list includes user names, email addresses, gender, addresses, mobile numbers, countries, states, and cities, as shown in the main section of the image.
Logout

Explanation: This option logs the user out of the system. It is a standard feature in web applications to ensure security and privacy.

![WhatsApp Image 2024-08-04 at 15 25 36_4956cdf7](https://github.com/user-attachments/assets/30c0e742-1d3d-4acc-9610-4613c8ddf408)

Model Types and Accuracy
The table lists various machine learning models along with their corresponding accuracy scores. Here are the models and their accuracies as shown in the table:

Artificial Neural Network (ANN)

Accuracy: 56.00000000000001
Explanation: ANNs are computing systems inspired by the biological neural networks that constitute animal brains. They are used for pattern recognition and classification tasks.
Naive Bayes

Accuracy: 63.0
Explanation: Naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features.
Support Vector Machine (SVM)

Accuracy: 60.5
Explanation: SVMs are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. They are effective in high-dimensional spaces.
Logistic Regression

Accuracy: 64.0
Explanation: Logistic regression is a statistical model that uses a logistic function to model a binary dependent variable. It's used for binary classification problems.
Gradient Boosting Classifier

Accuracy: 59.5
Explanation: Gradient Boosting is a machine learning technique for regression and classification problems, which builds a model in a stage-wise fashion from weak learners (usually decision trees).
Decision Tree Classifier

Accuracy: 60.5
Explanation: Decision Trees are a non-parametric supervised learning method used for classification and regression. They predict the value of a target variable by learning simple decision rules inferred from the data features.
K-Neighbors Classifier

Accuracy: 51.0
Explanation: The K-Nearest Neighbors (KNN) algorithm is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space.
Interpretation
Accuracy Scores: The accuracy score represents the percentage of correct predictions made by the model. Higher accuracy indicates better performance.
Comparison: Users can compare the performance of different models to choose the most effective one for predicting student profiles. For example, Logistic Regression has the highest accuracy at 64.0, while K-Neighbors Classifier has the lowest at 51.0.
Model Selection: Depending on the context and requirements, users might prioritize different models. For instance, a model with a slightly lower accuracy might be preferred if it is faster to train and predict.

![WhatsApp Image 2024-08-04 at 15 29 26_322befa6](https://github.com/user-attachments/assets/2875492a-e56c-44b5-a137-18ec6ed09324)


Bar Graph: The main part of the page displays a bar graph comparing the accuracy of different machine learning models in predicting student profiles:
X-axis: The models are listed: Artificial Neural Network-ANN, Naive Bayes, SVM, Logistic Regression, Gradient Boosting Classifier, Decision Tree Classifier, and KNeighbors Classifier.
Y-axis: Represents the accuracy of each model, ranging from 50% to 66%.
Bar Heights: Each bar shows the accuracy of a particular model, indicating its performance in predicting student profiles.

![WhatsApp Image 2024-08-04 at 15 41 49_0cf695a0](https://github.com/user-attachments/assets/f01b9af8-b5d5-4630-af2c-9f2ecd922b99)

![WhatsApp Image 2024-08-04 at 15 44 15_2675b975](https://github.com/user-attachments/assets/a7a9630f-f11a-4c29-87b1-aaf68c698a98)

![WhatsApp Image 2024-08-04 at 15 46 06_48582956](https://github.com/user-attachments/assets/c66a0249-977b-4115-8321-8740e82bc5ef)


This table shows the ratio of online student judgements for a given student profile. For example, 77.77% of the judgements for a certain student profile were "Excellent" and 22.22% of the judgements for that same student profile were "Poor".

![Screenshot 2024-08-04 134148](https://github.com/user-attachments/assets/70214503-98a4-45ab-896b-ae3782cd2e62)










