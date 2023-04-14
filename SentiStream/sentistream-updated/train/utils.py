from sklearn.metrics import accuracy_score

def polarity(label):
    return 1 if label == 1 else -1

def calculate_acc(data):
    y_true, y_pred = zip(*data)
    return accuracy_score(y_true, y_pred)
