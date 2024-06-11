#rf Class file

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from joblib import dump, load
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class RandomForestModel:
	def __init__(self):
	    self.model = None
	    self.columns = None


	def get_features_targets(self, final_rf_pd):

		# Split the data into features and target
		target = final_rf_pd['Classification'].astype(str)
		columns_to_select = ['Orientation', 'Brightness', 'Contrast', 'Hough_Info',
		                     'Harris_Corners', 'Contour_Info', 'Laplacian', 'SHV', 'Variance', 'Exposure', 'F_Stop', 'ISO',
		                     'Black_Pixels', 'Mid_tone_Pixels', 'White_Pixels', 'Faces', 'Eyes', 'Bodies', 'Focal_Length']
		features = final_rf_pd.loc[:, columns_to_select]
		features_encoded = pd.get_dummies(features, columns=['Orientation'])

		return features_encoded, target


	def random_forest_train(self,features, target):

		X_train, X_valid, y_train, y_valid = train_test_split(features, target, test_size=0.20, random_state=42)
		rf = RandomForestClassifier()
		rf.fit(X_train, y_train)

		# Set the trained model and columns
		self.model = rf
		self.columns = X_train.columns

		# Print or store any training metrics
		accuracy_train = rf.score(X_train, y_train)
		accuracy_valid = rf.score(X_valid, y_valid)
		print(f"Training Accuracy: {accuracy_train}")
		print(f"Validation Accuracy: {accuracy_valid}")


	def random_forest_test(self,features,target):
	    
		trained_classifier = self.model

		# features, target = self.get_features_targets()

		# Use the trained classifier to make predictions on the test data
		predictions = trained_classifier.predict(features)

		# ... rest of the code to print or record the predictions ...
		# ... rest of the code to print or record the predictions ...

		accuracy = accuracy_score(target, predictions)
		print(f"Test Accuracy: {accuracy}")

		self.generate_confusion_matrix(target, predictions)


	def generate_confusion_matrix(self, target, predictions):
	    # Create a confusion matrix
	    cm = confusion_matrix(target, predictions)
	    print(f"True Negative (TN): {cm[0, 0]}")
	    print(f"False Positive (FP): {cm[0, 1]}")
	    print(f"False Negative (FN): {cm[1, 0]}")
	    print(f"True Positive (TP): {cm[1, 1]}")

	    # Display the confusion matrix using seaborn
	    sns.heatmap(cm, annot=True, fmt='d')
	    plt.title('Confusion Matrix')
	    plt.xlabel('Predicted')
	    plt.ylabel('Actual')
	    plt.show()

	    # Generate the classification report
	    print(classification_report(target, predictions))















    # def test(self, features, target):
    #     if self.model is None:
    #         print("Error: Model not trained.")
    #         return

    #     predictions = self.model.predict(features)
    #     accuracy = accuracy_score(target, prediction