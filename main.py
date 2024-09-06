import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.activations import linear, sigmoid, relu
import matplotlib.pyplot as plt


def plt_samples(X,y):
    fig, axes = plt.subplots(5,5, figsize=(5,5))
    

    #choosing a random sample: (run several times to check different samples)
    np.random.seed(250)
    for ax in axes.flat:
        random_index = np.random.randint(0,5000, 1)
        X_sample = X[random_index]
        y_sample = int(y[random_index,0])
        X_reshaped = X_sample.reshape(20,20).T
        
        #displaying image of sample and labeled with its corresponding output
        ax.imshow(X_reshaped, cmap='gray')
        ax.set_title(f"{int(y_sample)}", fontsize=14)
        ax.axis("off")
          
    fig.suptitle("Labeled Samples")   
    plt.tight_layout(pad=0.5)
    

def plt_pred_samples(X,y, model):
    fig,axes = plt.subplots(5,5)
    np.random.seed(22)
     
    for ax in axes.flat:
        random_index = np.random.randint(5000)
        X_sample = X[random_index].reshape(20,20).T
        y_sample = int(y[random_index])
        f = model.predict(X[random_index].reshape(-1,400))
        yhat = np.argmax(f)
        ax.imshow(X_sample, cmap='gray')
        ax.set_title(f"{yhat} | {int(y_sample)}")
        ax.axis("off")

    fig.suptitle("Prediction | Actual")
    plt.tight_layout(pad=0.5)

    return None

def calculate_error(X,y,model):
    my_prediction = model.predict(X.reshape(-1,400))
    all_predictions = np.argmax(my_prediction, axis=1)
    error_count=0
    for i,target in enumerate(y):
        if all_predictions[i] != target:
            error_count += 1
    return error_count


'''Still in Development:
def plt_show_errors(X,y,model):
    my_prediction = model.predict(X.reshape(-1,400))
    all_predictions = np.argmax(my_prediction, axis=1)
    error_count=0
    wrong_predictions = []
    wrong_predictions_index = []
    for i,target in enumerate(y):
        if all_predictions[i] != target:
            error_count += 1
            wrong_predictions.append(all_predictions[i])
            wrong_predictions_index.append(i)
            
    size = len(wrong_predictions)
    fig,axes = plt.subplots(size/2,4, figsize=(2,2))
    plt.suptitle("Wrong Prediction | Correct Label")
    plt.tight_layout(pad=0.5)
    
    for i,ax in enumerate(axes.flat):
        ax.imshow(X[wrong_predictions_index], cmap='gray')
        ax.set_title(f"{wrong_predictions[i]} | {y[wrong_predictions_index]}")
    return None
'''
