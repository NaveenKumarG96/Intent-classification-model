# IntentClassification

#Create Virtual environment

```virtualenv venv```

#Activate the environment

```source venv/bin/activate```

Install the required dependencies into the virtual environment

```pip install -r requirements.txt```

Run the uvicorn standard command to run the app.py --- to run the model with lstm classifier

```uvicorn app:app --host 0.0.0.0 --port 8000```  

Use the browser and use this url for give the user query

```http://0.0.0.0:8000/```

Use Postman POST api to check the intent of the statment -  use the below url for post action.

```http://0.0.0.0:8000/intent```


# For running the model with roberta model.

Run the uvicorn standard command to run the appRoberta.py --- to run the model with Roberta model   NOTE : please include this file in root directory to run the appRoberta.py ```https://drive.google.com/file/d/1VbgF4KTjWyYTRzrzJymRxQPUBUSN-XgT/view?usp=sharing``` 

```uvicorn app:appRoberta --host 0.0.0.0 --port 8000```  



# Training the model -- for training purpose

```cd training```

```python3 training_code.py``` for training the model with LStm layers

```python3 robertabase.py``` for training the model with roberta as base line model.

custom_classifier.py contains the model with Lstm layers
robertabase.py contains the model with Roberta base.
