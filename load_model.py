import mlflow
import pandas as pd
import numpy as np

# TODO: Set tht MLFlow server uri
uri = "http://127.0.0.1:6001"
mlflow.set_tracking_uri(uri=uri)

# TODO: Provide model path/url
logged_model = 'runs:/2ab4a261b848489d981ddb5602500106/iris_model'

# Load model as a PyFuncModel.
loaded_model = mlflow.sklearn.load_model(logged_model)

# Input a random datapoint
data=np.array([[1.0,2.0,3.0,4.0]])

# TODO: Predict on a Pandas DataFrame. Due to the MLFlow functionality constrain.
#       The loaded model's predict function only accept dataframe as input instead of numpy array.
datapoint_df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3', 'feature4'])
prediction=loaded_model.predict(datapoint_df)

# Print out prediction result
print(prediction)