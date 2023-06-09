def main():
    import wandb
    from wandb.keras import WandbCallback
    wandb.init(project="wandb_test_iris", entity='rapry60', settings=wandb.Settings(_disable_stats=True))
    # we need to disable stats [CLI] Breaking: wandb.init() crashes (BSOD)
    #wandb.init(project="wandb_test_iris")
    wandb.config = {
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 128
    }
    import numpy as np
    from keras.layers import Dense
    from keras.models import Sequential
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from tensorflow.keras.optimizers import Adam
    iris_data = load_iris() # load the iris dataset
    print('Example data: ')
    print(iris_data.data[:5])
    print('Example labels: ')
    print(iris_data.target[:5])
    x = iris_data.data
    y_ = iris_data.target.reshape(-1, 1) # Convert data to a single column
    # One Hot encode the class labels
    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(y_)
    #print(y)
    # Split the data for training and testing
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)
    # Build the model
    model = Sequential()
    model.add(Dense(10, input_shape=(4,), activation='relu', name='fc1'))
    model.add(Dense(10, activation='relu', name='fc2'))
    model.add(Dense(3, activation='softmax', name='output'))
    # Adam optimizer with learning rate of 0.001
    optimizer = Adam(lr=0.001)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print('Neural Network Model Summary: ')
    print(model.summary())
    # Train the model
    model.fit(train_x, train_y, verbose=2, batch_size=5, epochs=200, callbacks=[WandbCallback()])
    # Test on unseen data
    results = model.evaluate(test_x, test_y)
    print('Final test set loss: {:4f}'.format(results[0]))
    print('Final test set accuracy: {:4f}'.format(results[1]))
    return
if __name__ == '__main__':
    main()
