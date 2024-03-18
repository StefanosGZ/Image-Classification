import keras.optimizers
import matplotlib.pyplot as plt
import numpy as np
import cv2
import copy
from sklearn.model_selection import train_test_split
import imutils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Rescaling, RandomFlip, RandomCrop, RandomZoom, RandomBrightness,RandomContrast, RandomTranslation
import keras
from sklearn import metrics
import tensorflow as tf
from imblearn.over_sampling import SMOTE
import seaborn as sns



# Define a custom F1 score metric
def balanced_accuracy(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.equal(y_true, 1), tf.equal(tf.round(y_pred), 1)), dtype=tf.float32))
    true_negatives = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.equal(y_true, 0), tf.equal(tf.round(y_pred), 0)), dtype=tf.float32))
    false_positives = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.equal(y_true, 0), tf.equal(tf.round(y_pred), 1)), dtype=tf.float32))
    false_negatives = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.equal(y_true, 1), tf.equal(tf.round(y_pred), 0)), dtype=tf.float32))

    sensitivity = true_positives / (true_positives + false_negatives + tf.keras.backend.epsilon())
    specificity = true_negatives / (true_negatives + false_positives + tf.keras.backend.epsilon())

    balanced_acc = (sensitivity + specificity) / 2

    return balanced_acc

def No_Oversampling(XTrain, YTrain):
    XTrain = XTrain.reshape(-1,28,28,3)
    return XTrain, YTrain
def Custom_oversampling(XTrain,YTrain):
    nevu_list, melanoma_list = [], []
    i = 0

    # Make lists consisting of the two different classes
    while i < len(YTrain):
        if YTrain[i] == 0.0:
            nevu_list.append(XTrain[i])
        else:
            melanoma_list.append(XTrain[i])
        i += 1

    # Turning into picture
    Nevus = np.reshape(nevu_list, (len(nevu_list), 28, 28, 3))
    Melanomas = np.reshape(melanoma_list, (len(melanoma_list), 28, 28, 3))


    Temporary_Melanomas = copy.deepcopy(Melanomas)
    Temporary_Melanomas = list(Temporary_Melanomas)
    # Over sampling: ii=1-3 different orientation, ii=4,5 mirrored picture
    for ii in range(1, 6):
        for Melanoma in Melanomas:
            if ii < 4:
                angle_ = ii * 90
                rotate_picture = imutils.rotate(Melanoma, angle=angle_)
                Temporary_Melanomas.append(rotate_picture)
            else:
                if ii == 4:
                    q = 0
                else:
                    q == 1
                mirrored_image = cv2.flip(Melanoma, q)
                Temporary_Melanomas.append(mirrored_image)

    # Crate new X_train and Y_train with the oversampling datapoints
    New_XTrain = np.concatenate((Nevus, Temporary_Melanomas), axis=0)
    New_YTrain = [0] * len(Nevus) + [1] * len(Temporary_Melanomas)

    # Shuffling the datas, so that the connection is not lost
    Shuffle = list(zip(New_XTrain, New_YTrain))
    np.random.shuffle(Shuffle)
    New_XTrain, New_YTrain = zip(*Shuffle)
    return New_XTrain, New_YTrain

def Normal_oversampling(XTrain,YTrain):
    nevu_list, melanoma_list = [], []
    i = 0

    # Make lists consisting of the two different classes
    while i < len(YTrain):
        if YTrain[i] == 0.0:
            nevu_list.append(XTrain[i])
        else:
            melanoma_list.append(XTrain[i])
        i += 1

    # Turning into picture
    Nevus = np.reshape(nevu_list, (len(nevu_list), 28, 28, 3))
    Melanomas = np.reshape(melanoma_list, (len(melanoma_list), 28, 28, 3))
    Melanomas = list(Melanomas)
    for i in range(len(Nevus) - len(Melanomas)):
        Melanomas.append(Melanomas[i])

    # Crate new X_train and Y_train with the oversampling datapoints
    New_XTrain = np.concatenate((Nevus, Melanomas), axis=0)
    New_YTrain = [0] * len(Nevus) + [1] * len(Melanomas)

    # Shuffling the datas, so that the connection is not lost
    Shuffle = list(zip(New_XTrain, New_YTrain))
    np.random.shuffle(Shuffle)
    New_XTrain, New_YTrain = zip(*Shuffle)
    return New_XTrain, New_YTrain


def Shifted_OverSampling(XTrain, YTrain, augmentation_factor=4):
    nevu_list, melanoma_list = [], []
    i = 0

    # Make lists consisting of the two different classes
    while i < len(YTrain):
        if YTrain[i] == 0.0:
            nevu_list.append(XTrain[i])
        else:
            melanoma_list.append(XTrain[i])
        i += 1

    # Turning into picture
    Nevus = np.reshape(nevu_list, (len(nevu_list), 28, 28, 3))
    Melanomas = np.reshape(melanoma_list, (len(melanoma_list), 28, 28, 3))

    Temporary_Melanomas = copy.deepcopy(Melanomas)
    Temporary_Melanomas = list(Temporary_Melanomas)

    # Over-sampling: Apply random shifts and augment images
    for _ in range(augmentation_factor):
        for Melanoma in Melanomas:
            # Define random shift values for both horizontal and vertical directions
            dx, dy = np.random.randint(-2, 3, 2)  # Adjust the range as needed

            # Apply the shift to the image
            shifted_image = np.roll(Melanoma, (dx, dy), axis=(0, 1))
            Temporary_Melanomas.append(shifted_image)

    # Create new X_train and Y_train with the oversampling datapoints
    New_XTrain = np.concatenate((Nevus, Temporary_Melanomas), axis=0)
    New_YTrain = [0] * len(Nevus) + [1] * len(Temporary_Melanomas)

    # Shuffle the data to prevent bias
    Shuffle = list(zip(New_XTrain, New_YTrain))
    np.random.shuffle(Shuffle)
    New_XTrain, New_YTrain = zip(*Shuffle)

    return New_XTrain, New_YTrain


def Zoomed_OverSampling(XTrain, YTrain, augmentation_factor=4, zoom_range=(0.9, 1.1), target_size=(28, 28)):
    nevu_list, melanoma_list = [], []
    i = 0

    # Make lists consisting of the two different classes
    while i < len(YTrain):
        if YTrain[i] == 0.0:
            nevu_list.append(XTrain[i])
        else:
            melanoma_list.append(XTrain[i])
        i += 1

    # Turning into picture
    Nevus = np.reshape(nevu_list, (len(nevu_list), 28, 28, 3))
    Melanomas = np.reshape(melanoma_list, (len(melanoma_list), 28, 28, 3))

    Temporary_Melanomas = copy.deepcopy(Melanomas)
    Temporary_Melanomas = list(Temporary_Melanomas)

    # Over-sampling: Apply random zooms and augment images
    for _ in range(augmentation_factor):
        for Melanoma in Melanomas:
            zoom_factor = np.random.uniform(zoom_range[0], zoom_range[1])
            zoomed_image = cv2.resize(Melanoma, None, fx=zoom_factor, fy=zoom_factor)
            zoomed_image = cv2.resize(zoomed_image, target_size)  # Resize to a consistent size
            Temporary_Melanomas.append(zoomed_image)

    # Create new X_train and Y_train with the oversampling datapoints
    New_XTrain = np.concatenate((Nevus, Temporary_Melanomas), axis=0)
    New_YTrain = [0] * len(Nevus) + [1] * len(Temporary_Melanomas)

    # Shuffle the data to prevent bias
    Shuffle = list(zip(New_XTrain, New_YTrain))
    np.random.shuffle(Shuffle)
    New_XTrain, New_YTrain = zip(*Shuffle)

    return New_XTrain, New_YTrain

def Smote(XTrain, YTrain):
    smote = SMOTE(random_state=42)

    # Reshape XTrain if needed
    XTrain = XTrain.reshape(-1, 28 * 28 * 3)

    # Apply SMOTE to the training data
    XTrain_resampled, YTrain_resampled = smote.fit_resample(XTrain, YTrain)

    # Reshape XTrain_resampled back to the original shape
    XTrain_resampled = XTrain_resampled.reshape(-1, 28, 28, 3)

    return XTrain_resampled, YTrain_resampled
def main():
    XTrain = np.load("XTrain_Classification1.npy")
    YTrain = np.load("ytrain_Classification1.npy")
    XTest = np.load("Xtest_Classification1 (1).npy")
    i = 0
    nevvu = 0
    mella = 0

    #Divide into training and validation sets
    XTrain, XVal, YTrain, YVal = train_test_split(XTrain, YTrain, test_size=0.15, random_state=22, stratify=YTrain)
    XVal = np.reshape(XVal, (len(XVal), 28, 28, 3))
    XTest = np.reshape(XTest, (len(XTest), 28, 28, 3))
    nevvu = 0
    mella = 0

    New_XTrain, New_YTrain = Custom_oversampling(XTrain, YTrain)
    #New_XTrain, New_YTrain = No_Oversampling(XTrain, YTrain)
    #New_XTrain, New_YTrain = Normal_oversampling(XTrain, YTrain)
    #New_XTrain, New_YTrain = Shifted_OverSampling(XTrain, YTrain, augmentation_factor=4)
    #New_XTrain, New_YTrain = Zoomed_OverSampling(XTrain, YTrain, augmentation_factor=4, zoom_range=(0.7, 1.3))
    #New_XTrain, New_YTrain = Smote(XTrain, YTrain)

    New_XTrain = np.array(New_XTrain)
    XVal = np.array(XVal)
    New_YTrain = np.array(New_YTrain)
    YVal = np.array(YVal)

    #Sequential model creation
    model = Sequential()
    #model.add(RandomBrightness(factor=0.2))
    model.add(RandomZoom(height_factor=(-0.2,0.3),width_factor=(-0.2,0.3)))
    #model.add(RandomTranslation(height_factor=(-0.2,0.3),width_factor=(-0.2,0.3)))
    model.add(Rescaling(1/255))
    #Convolutional layers
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    #Flattening layer
    model.add(Flatten())

    #Fully conncected layers, with regularization
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    #Binary classification
    model.add(Dense(1, activation='sigmoid'))

    #Creates a file where the best model will be saved
    model_optimal_callback = keras.callbacks.ModelCheckpoint(
        filepath="Optimal/",
        save_weights_only=True,
        monitor='val_balanced_accuracy',
        mode='max',
        save_best_only=True)

    #Choosing an optimizer for the model
    Optimizer = keras.optimizers.Adam(learning_rate=0.001)

    #Creating the model
    model.compile(optimizer=Optimizer, loss='binary_crossentropy', metrics=['acc', balanced_accuracy])

    #Model training and validation
    class_weights = {0:1, 1:2}  # Example: Assign higher weight to class 1
    Training = model.fit(New_XTrain, New_YTrain, batch_size=64, epochs=30, validation_data=(XVal, YVal), verbose=1,
                         callbacks=model_optimal_callback, class_weight=class_weights)

    #Loads the best model
    model.load_weights("Optimal/")

    #Predicts the Validation to better understand the model's performance
    Ypred = model.predict(XVal)

    Ypred[Ypred < 0.5] = 0
    Ypred[Ypred >= 0.5] = 1

    Ypred = np.reshape(Ypred, (len(Ypred), 1))

    # Calculate balanced accuracy for class 1
    Balanced_accuracy = metrics.balanced_accuracy_score(YVal, Ypred)
    print(f"Balanced accuracy: {Balanced_accuracy}")

    Confusion_matrix = metrics.confusion_matrix(YVal,Ypred)
    print(f"Confusion Matrix:\n {Confusion_matrix}")
    TP = int(Confusion_matrix[0][0])
    FP = int(Confusion_matrix[0][1])
    FN = int(Confusion_matrix[1][0])
    TN = int(Confusion_matrix[1][1])

    FP_percentage = FP/(TP+FP)
    FN_percentae = FN/(FN+TN)

    print(f"False positive: {FP_percentage}")
    print(f"False negative: {FN_percentae}")

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(Training.history['loss'], label='Training loss')
    plt.plot(Training.history['val_loss'], label='Validation loss')
    plt.plot(Training.history['balanced_accuracy'], label='Training balanced accuracy')
    plt.plot(Training.history['val_balanced_accuracy'], label='Validation balanced accuracy')
    plt.plot(Training.history['acc'], label='Training accuracy')
    plt.plot(Training.history['val_acc'], label='Validation accuracy')
    plt.xlabel("Epochs")
    plt.legend(loc="best", bbox_to_anchor=(1, 1))
    plt.show()
    plt.figure(figsize=(6, 4))
    sns.heatmap(Confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    YTest = model.predict(XTest)

    YTest[YTest < 0.5] = 0
    YTest[YTest >= 0.5] = 1

    YTest = np.ndarray.flatten(YTest)

    np.save("Assignment 3.3 Submission", YTest)

    YY = np.load("Assignment 3.3 Submission.npy")
    q = 0
    for Y in YY:
        if Y != 0. or Y != 1.:
            q = 1
    if q == 0:
        print("All good")
    print(YTest)
main()