from sklearn.metrics import confusion_matrix
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_loss_curve(model_history):
    fig = plt.figure()
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    #fig.savefig("loss.png")

def plot_acc_curve(model_history):
    fig = plt.figure()
    plt.plot(model_history.history['acc'])
    plt.plot(model_history.history['val_acc'])
    plt.title('model acc')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    #fig.savefig("acc.png")

def plot_confusion_matrix(model, X_test, y_test, labels, lb):
    def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
        df_cm = pd.DataFrame(
            confusion_matrix, index=class_names, columns=class_names, 
        )
        fig = plt.figure(figsize=figsize)
        try:
            heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")

        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        #fig.savefig("confusion_matrix.png")


    preds = model.predict(X_test, 
                                batch_size=16, 
                                verbose=2)
    preds=preds.argmax(axis=1)
    preds = preds.astype(int).flatten()
    preds = (lb.inverse_transform((preds)))

    actual = y_test.argmax(axis=1)
    actual = actual.astype(int).flatten()
    actual = (lb.inverse_transform((actual)))

    classes = labels
    classes.sort()    

    c = confusion_matrix(actual, preds)
    print_confusion_matrix(c, class_names = classes)


