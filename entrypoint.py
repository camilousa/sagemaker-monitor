import joblib
import os
import json
from sagemaker_containers.beta.framework import (content_types, encoders, env, modules, transformer, worker)

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.jobli"))
    return clf

def predict_fn(input_data, model):
     if len(input_data.shape) == 1:
      proba = model.predict_proba(input_data.reshape(-1, 1).T)
     else:
      proba = model.predict_proba(input_data)   
     return [proba[0][1]]


def output_fn(prediction, accept):

    if accept == "application/json":
        return worker.Response(json.dumps(prediction[0]), mimetype=accept)
    elif accept == 'text/csv':
        #return worker.Response(encoders.encode(prediction[0], accept), mimetype=accept)
        return worker.Response(json.dumps(prediction[0]), mimetype=accept)
