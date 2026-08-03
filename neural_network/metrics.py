import numpy as np
import matplotlib.pyplot as plt

def accuracy(model, X, y):
    correct = sum(np.argmax(model.predict(x)) == np.argmax(y) 
                  for x, y in zip(X, y))
    return correct / len(X) * 100

def confusion_matrix(model, X, y):
    matrix = np.zeros((10, 10), dtype=int)
    for x, true in zip(X, y):
        pred = np.argmax(model.predict(x))
        matrix[np.argmax(true)][pred] += 1
    return matrix

def plot_losses(losses):
    train = losses["train"]
    val = losses["val"]
    plt.plot(range(1, len(train) + 1), train, label = "Train", marker='o')
    if len(val) > 0:
        plt.plot(range(1, len(val) + 1), val, label = "Validation", marker='o')
    plt.xticks(range(1, len(train) + 1))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()

def plot_confusion_matrix(matrix):
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(range(10))
    plt.yticks(range(10))
    
    for i in range(10):
        for j in range(10):
            plt.text(j, i, matrix[i][j], ha='center', va='center')
    
    plt.show()

def summary(model, X_test, y_test):
    acc = accuracy(model, X_test, y_test)
    matrix = confusion_matrix(model, X_test, y_test)
    print(f"Test Accuracy: {acc:.2f}%")
    plot_confusion_matrix(matrix)