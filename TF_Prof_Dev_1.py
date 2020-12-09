# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 19:18:16 2020

@author: Steven

Copied, augmented, tweaked, created code from Coursera courses: 
    DeepLearning.AI TensorFlow Developer
        Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning
        Convolutional Neural Networks in TensorFlow
            CatsVsDogs_CNN_1()
            HumansVsHorses_NN_1()
            RockPaperScissors_NN_1()
            MINIST_signs_1()
        Natural Language Processing in TensorFlow
            tokenize_NLP_1()
            tokenize_NLP_2()
            IMDB_NLP_1()
            sarcasm_NLP_1()
            overfitting_NLP_1()
            shakespeare_NLP_1()
        Sequences, Time Series and Prediction
            season_seq_1()
            season_seq_2()
            season_seq_3()
            sunspot_seq_1()
            sunspot_seq_2()
"""

import csv
import json
import matplotlib.pyplot as plt
import matplotlib.image  as mpimg
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import zipfile


### Classify images of cats and dogs using a CNN with input augmentation ###
def CatsVsDogs_CNN_1():

    import zipfile
    import shutil
    from tensorflow.keras.optimizers import RMSprop
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from shutil import copyfile
    from os import getcwd
    
    path_cats_and_dogs = f"{getcwd()}/Data/cats-and-dogs.zip"
    shutil.rmtree('/tmp_catdog')
    
    local_zip = path_cats_and_dogs
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('/tmp_catdog')
    zip_ref.close()
    
    print("The number of cat and dog images are: ")
    print(len(os.listdir('/tmp_catdog/PetImages/Cat/')))
    print(len(os.listdir('/tmp_catdog/PetImages/Dog/')))

    # Use os.mkdir to create your directories
    # You will need a directory for cats-v-dogs, and subdirectories for training
    # and testing. These in turn will need subdirectories for 'cats' and 'dogs'
    try:
        base_dir = '/tmp_catdog/cats-v-dogs'
    
        train_dir = os.path.join(base_dir, 'training')
        validation_dir = os.path.join(base_dir, 'testing')
    
        # Directory with our training cat/dog pictures
        train_cats_dir = os.path.join(train_dir, 'cats')
        train_dogs_dir = os.path.join(train_dir, 'dogs')
    
        # Directory with our validation cat/dog pictures
        validation_cats_dir = os.path.join(validation_dir, 'cats')
        validation_dogs_dir = os.path.join(validation_dir, 'dogs')
        
        os.makedirs(train_cats_dir)
        os.makedirs(train_dogs_dir)
        os.makedirs(validation_dogs_dir)
        os.makedirs(validation_cats_dir)
        print("Subdirectories Created")
        
    except OSError:
        print("Subdirectories NOT Created!!")
     
        
    # split_data which takes a SOURCE directory, a TRAINING directory, a 
    # TESTING directory, a SPLIT SIZE.
    # Files randomized, checked for a zero file length
    # Ex) if SOURCE = PetImages/Cat, SPLIT SIZE = .9, 90% of the images in 
    # PetImages/Cat will be copied to the TRAINING dir
    def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
        list=os.listdir(SOURCE)
        list=random.sample(list, len(list))
        for i,name in enumerate(list):
            source=SOURCE+name
            print(source)
            if i < SPLIT_SIZE*len(list):
                copyfile(source, TRAINING+name)
                print(i)
            else:
                copyfile(source, TESTING+name)
    
    
    CAT_SOURCE_DIR = "/tmp_catdog/PetImages/Cat/"
    TRAINING_CATS_DIR = "/tmp_catdog/cats-v-dogs/training/cats/"
    TESTING_CATS_DIR = "/tmp_catdog/cats-v-dogs/testing/cats/"
    DOG_SOURCE_DIR = "/tmp_catdog/PetImages/Dog/"
    TRAINING_DOGS_DIR = "/tmp_catdog/cats-v-dogs/training/dogs/"
    TESTING_DOGS_DIR = "/tmp_catdog/cats-v-dogs/testing/dogs/"
    
    split_size = .9
    split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
    split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

    print(len(os.listdir('/tmp_catdog/cats-v-dogs/training/cats/')))
    print(len(os.listdir('/tmp_catdog/cats-v-dogs/training/dogs/')))
    print(len(os.listdir('/tmp_catdog/cats-v-dogs/testing/cats/')))
    print(len(os.listdir('/tmp_catdog/cats-v-dogs/testing/dogs/')))

    
    # KERAS MODEL TO CLASSIFY CATS V DOGS
    model = tf.keras.models.Sequential([
               # This is the first convolution
                tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
                tf.keras.layers.MaxPooling2D(2, 2),
                # The second convolution
                tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2,2),
                # The third convolution
                tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2,2),
                # Flatten the results to feed into a DNN
                tf.keras.layers.Flatten(),
                # 512 neuron hidden layer
                tf.keras.layers.Dense(512, activation='relu'),
                # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
                tf.keras.layers.Dense(1, activation='sigmoid')])
                
    model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
    
    
    TRAINING_DIR = '/tmp_catdog/cats-v-dogs/training/'
    train_datagen = ImageDataGenerator(
        rescale=1/255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    
    # NOTE: YOU MUST USE A BATCH SIZE OF 10 (batch_size=10) FOR THE 
    # TRAIN GENERATOR.
    train_generator = train_datagen.flow_from_directory(
            TRAINING_DIR,  # This is the source directory for training images
            target_size=(150, 150),  # All images will be resized to 150x150
            batch_size=10,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='binary')
    
    VALIDATION_DIR = '/tmp_catdog/cats-v-dogs/testing/'
    validation_datagen = ImageDataGenerator(
        rescale=1/255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    
    # NOTE: YOU MUST USE A BACTH SIZE OF 10 (batch_size=10) FOR THE 
    # VALIDATION GENERATOR.
    validation_generator = validation_datagen.flow_from_directory(
            VALIDATION_DIR,  # This is the source directory for training images
            target_size=(150, 150),  # All images will be resized to 150x150
            batch_size=10,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='binary')
    
    history = model.fit_generator(train_generator,
                                  epochs=2,
                                  verbose=1,
                                  validation_data=validation_generator)    
    
    # PLOT LOSS AND ACCURACY
    import matplotlib.image  as mpimg
    import matplotlib.pyplot as plt
    
    #-----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    #-----------------------------------------------------------
    acc=history.history['acc']
    val_acc=history.history['val_acc']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    
    epochs=range(len(acc)) # Get number of epochs
    
    #------------------------------------------------
    # Plot training and validation accuracy per epoch
    #------------------------------------------------
    plt.plot(epochs, acc, 'r', "Training Accuracy")
    plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
    plt.title('Training and validation accuracy')
    plt.figure()
    
    #------------------------------------------------
    # Plot training and validation loss per epoch
    #------------------------------------------------
    plt.plot(epochs, loss, 'r', "Training Loss")
    plt.plot(epochs, val_loss, 'b', "Validation Loss")
    
    
    plt.title('Training and validation loss')
#CatsVsDogs_CNN_1()

### Classifying Hourses and Humans using Pre-Trained Network ###
def HumansVsHorses_NN_1():
    # PreTrained Netowrk found here:
    # https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \

    path_inception = f"{os.getcwd()}/Data/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

    # Import the inception model  
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    
    # Create an instance of the inception model from the local pre-trained weights
    local_weights_file = path_inception
    
    pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                    include_top = False, 
                                    weights = None)
    
    pre_trained_model.load_weights(local_weights_file)
    
    # Make all the layers in the pre-trained model non-trainable
    for layer in pre_trained_model.layers:
      layer.trainable = False
      
    # Print the model summary
    pre_trained_model.summary()

    
    last_layer = pre_trained_model.get_layer('mixed7')
    print('last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output

    
    # Define a Callback class that stops training once accuracy reaches 97.0%
    class myCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.97):
          print("\nReached 97.0% accuracy so cancelling training!")
          self.model.stop_training = True


    from tensorflow.keras.optimizers import RMSprop
    
    # Flatten the output layer to 1 dimension
    x = tf.keras.layers.Flatten()(last_output)
    # Add a fully connected layer with 1,024 hidden units and ReLU activation
    x = tf.keras.layers.Dense(1024,activation='relu')(x)
    # Add a dropout rate of 0.2
    x = tf.keras.layers.Dropout(0.2)(x)                  
    # Add a final sigmoid layer for classification
    x = tf.keras.layers.Dense  (1, activation='sigmoid')(x)           
    
    model = tf.keras.Model(pre_trained_model.input, x) 
    
    model.compile(optimizer = RMSprop(lr=0.0001), 
                  loss = 'binary_crossentropy', 
                  metrics = ['accuracy'])
    
    model.summary()


    # Get the Horse or Human dataset
    path_horse_or_human = f"{os.getcwd()}/Data/horse-or-human.zip"
    # Get the Horse or Human Validation dataset
    path_validation_horse_or_human = f"{os.getcwd()}/Data/validation-horse-or-human.zip"
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import zipfile
    import shutil
    
    shutil.rmtree('/tmp_humanhorse')
    local_zip = path_horse_or_human
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('/tmp_humanhorse/training')
    zip_ref.close()
    
    local_zip = path_validation_horse_or_human
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('/tmp_humanhorse/validation')
    zip_ref.close()


    # Define our example directories and files
    train_dir = '/tmp_humanhorse/training'
    validation_dir = '/tmp_humanhorse/validation'
    
    train_horses_dir = os.path.join(train_dir,'horses')
    train_humans_dir = os.path.join(train_dir,'humans')
    validation_horses_dir = os.path.join(validation_dir,'horses')
    validation_humans_dir = os.path.join(validation_dir,'humans')
    
    train_horses_fnames = os.listdir(train_horses_dir)
    train_humans_fnames = os.listdir(train_humans_dir)
    validation_horses_fnames = os.listdir(validation_horses_dir)
    validation_humans_fnames = os.listdir(validation_humans_dir)
    
    print("The number of horses/humans for training/validatoin: ")
    print(len(train_horses_fnames))
    print(len(train_humans_fnames))
    print(len(validation_horses_fnames))
    print(len(validation_humans_fnames))


    # Add our data-augmentation parameters to ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       rotation_range = 40,
                                       width_shift_range = 0.2,
                                       height_shift_range = 0.2,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = True)
    
    # Note that the validation data should not be augmented!
    test_datagen = ImageDataGenerator(rescale = 1./255,
                                       rotation_range = 40,
                                       width_shift_range = 0.2,
                                       height_shift_range = 0.2,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = True)
    
    # Flow training images in batches of 20 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        batch_size=20,
                                                        class_mode='binary',
                                                        target_size=(150,150))
    
    # Flow validation images in batches of 20 using test_datagen generator
    validation_generator =  test_datagen.flow_from_directory( validation_dir,
                                                        batch_size=20,
                                                        class_mode='binary',
                                                        target_size=(150,150))


    # Run this and see how many epochs it should take before the callback
    # fires, and stops training at 97% accuracy
    
    callbacks = myCallback()
    history = model.fit_generator(train_generator,
              steps_per_epoch=10,  
              epochs=3,
              verbose=1,
              validation_steps=8,
              validation_data=validation_generator,
              callbacks=[callbacks])


    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))
    
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()
        
    plt.show()
#HumansVsHorses_NN_1()

### Determines if computer generated hands are either rock/paper/scissors ###
def RockPaperScissors_NN_1():
    # https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip
    # https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip

    local_zip = f"{os.getcwd()}/Data/rps.zip"
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('/tmp/')
    zip_ref.close()
    
    local_zip = f"{os.getcwd()}/Data/rps-test-set.zip"
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('/tmp/')
    zip_ref.close()    

    rock_dir = os.path.join('/tmp/rps/rock')
    paper_dir = os.path.join('/tmp/rps/paper')
    scissors_dir = os.path.join('/tmp/rps/scissors')
    
    
    print('total training rock images:', len(os.listdir(rock_dir)))
    print('total training paper images:', len(os.listdir(paper_dir)))
    print('total training scissors images:', len(os.listdir(scissors_dir)))
    
    print("List 10 images in each group:")
    rock_files = os.listdir(rock_dir)
    print(rock_files[:10])
    paper_files = os.listdir(paper_dir)
    print(paper_files[:10])
    scissors_files = os.listdir(scissors_dir)
    print(scissors_files[:10])


    import matplotlib.image as mpimg
    
    pic_index = 2
    next_rock = [os.path.join(rock_dir, fname) 
                    for fname in rock_files[pic_index-2:pic_index]]
    next_paper = [os.path.join(paper_dir, fname) 
                    for fname in paper_files[pic_index-2:pic_index]]
    next_scissors = [os.path.join(scissors_dir, fname) 
                    for fname in scissors_files[pic_index-2:pic_index]]
    
    for i, img_path in enumerate(next_rock+next_paper+next_scissors):
        #print(img_path)
        img = mpimg.imread(img_path)
        plt.imshow(img)
        plt.axis('Off')
        plt.show()
      
      
    import keras_preprocessing
    from keras_preprocessing import image
    from keras_preprocessing.image import ImageDataGenerator
    
    TRAINING_DIR = "/tmp/rps/"
    training_datagen = ImageDataGenerator(
          rescale = 1./255,
    	    rotation_range=40,
          width_shift_range=0.2,
          height_shift_range=0.2,
          shear_range=0.2,
          zoom_range=0.2,
          horizontal_flip=True,
          fill_mode='nearest')
    
    VALIDATION_DIR = "/tmp/rps-test-set/"
    validation_datagen = ImageDataGenerator(rescale = 1./255)
    
    train_generator = training_datagen.flow_from_directory(
    	TRAINING_DIR,
    	target_size=(150,150),
    	class_mode='categorical',
      batch_size=126
    )
    
    validation_generator = validation_datagen.flow_from_directory(
    	VALIDATION_DIR,
    	target_size=(150,150),
    	class_mode='categorical',
      batch_size=126
    )
    
    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 150x150 with 3 bytes color
        # This is the first convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # The third convolution
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # The fourth convolution
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    
    
    model.summary()
    model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    history = model.fit(train_generator, epochs=25, steps_per_epoch=20, validation_data = validation_generator, verbose = 1, validation_steps=3)
    model.save("rps.h5")

      
    ### Plotting Accuracy ###
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))
    
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()
     
    plt.show()      
#RockPaperScissors_NN_1()

### MINST Sign Language ###
def MINIST_signs_1():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    def get_data(filename):
      # The first line contains the column headers, ignore it
      # Each successive line contians 785 comma separated values between 0 and 255
      # The first value is the label
      # The rest are the pixel values for that picture
      # The function will return 2 np.array types. One with all the labels
      # One with all the images

        with open(filename) as training_file:
            reader=csv.reader(training_file)
            
            if GG:
                maxval=27455
                images=np.zeros((maxval,28,28))
                labels=np.zeros((maxval))
            else:
                maxval=7172
                images=np.zeros((maxval,28,28))
                labels=np.zeros((maxval))
            
            for i,txt in enumerate(reader):
                if i == 0:
                    pass
                elif i <= maxval:
                    label=np.array(txt[0]).astype(float)
                    labels[i-1]=label
                    image=np.array(txt[1:785]).astype(float)
                    images[i-1,:,:]=np.array_split(image,28)
                else:
                    pass
                
        return images, labels
    
    path_sign_mnist_train = f"{os.getcwd()}/Data/sign_mnist_train.csv"
    path_sign_mnist_test = f"{os.getcwd()}/Data/sign_mnist_test.csv"
    GG=1
    training_images, training_labels = get_data(path_sign_mnist_train)
    GG=0
    testing_images, testing_labels = get_data(path_sign_mnist_test)
    
    # Keep these
    print(training_images.shape)
    print(training_labels.shape)
    print(testing_images.shape)
    print(testing_labels.shape)


    # add another dimension to the data
    # EX) if your array is (10000, 28, 28) -> (10000, 28, 28, 1)
        
    training_images = np.expand_dims(training_images, axis=4)
    testing_images = np.expand_dims(testing_images, axis=4)
    
    # Create an ImageDataGenerator and do Image Augmentation
    train_datagen = ImageDataGenerator(
                    rescale = 1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest')
    
    
    validation_datagen = ImageDataGenerator(
                rescale = 1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest')
        
    # Keep These
    print(training_images.shape)
    print(testing_images.shape)


    # Define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (2,2), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(64, (2,2), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(26, activation='softmax')
    ])
    
    model.summary()
    
    # Compile Model. 
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    # Train the Model
    history = model.fit_generator(train_datagen.flow(training_images,training_labels), epochs=15,steps_per_epoch=550,
                                  validation_data = validation_datagen.flow(testing_images,testing_labels))
    
    model.evaluate(testing_images, testing_labels, verbose=0)


    # Plot the chart for accuracy and loss on both training and validation
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))
    
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    
    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()
#MINIST_signs_1()

### Tokenizes a paragraph and uses NLP ###
def tokenize_NLP_1():
    # https://storage.googleapis.com/laurencemoroney-blog.appspot.com/bbc-text.csv
    
    import csv
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    
    # Stopwords list from https://github.com/Yoast/YoastSEO.js/blob/develop/src/config/stopwords.js
    stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]


    sentences = []
    labels = []
    with open(f"{os.getcwd()}/Data/bbc-text.csv", 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            labels.append(row[0])
            sentence = row[1]
            for word in stopwords:
                token = " " + word + " "
                sentence = sentence.replace(token, " ")
                sentence = sentence.replace("  ", " ")
            sentences.append(sentence)
    
    print(len(sentences))
    print(sentences[0])
    

    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    print(len(word_index))

    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, padding='post')
    print(padded[0])
    print(padded.shape)

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)
    label_word_index = label_tokenizer.word_index
    label_seq = label_tokenizer.texts_to_sequences(labels)
    print(label_seq)
    print(label_word_index)
#tokenize_NLP_1()

### Tokenizes a paragraph and uses NLP ###
def tokenize_NLP_2():
    # https://storage.googleapis.com/laurencemoroney-blog.appspot.com/bbc-text.csv
    
    import csv
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_portion = .8
    
    sentences = []
    labels = []
    stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
    print(len(stopwords))
    
    with open(f"{os.getcwd()}/Data/bbc-text.csv", 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            labels.append(row[0])
            sentence = row[1]
            for word in stopwords:
                token = " " + word + " "
                sentence = sentence.replace(token, " ")
            sentences.append(sentence)
    
    print(len(labels))
    print(len(sentences))
    print(sentences[0])
    
    
    train_size = int(len(sentences) * training_portion)
    
    train_sentences = sentences[:train_size]
    train_labels = labels[:train_size]
    
    validation_sentences = sentences[train_size:]
    validation_labels = labels[train_size:]
    
    print(train_size)
    print(len(train_sentences))
    print(len(train_labels))
    print(len(validation_sentences))
    print(len(validation_labels))


    tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_sentences)
    word_index = tokenizer.word_index
    
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)
    
    print(len(train_sequences[0]))
    print(len(train_padded[0]))
    
    print(len(train_sequences[1]))
    print(len(train_padded[1]))
    
    print(len(train_sequences[10]))
    print(len(train_padded[10]))


    validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
    validation_padded = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length)
    
    print(len(validation_sequences))
    print(validation_padded.shape)


    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)
    
    training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
    validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))
    
    print(training_label_seq[0])
    print(training_label_seq[1])
    print(training_label_seq[2])
    print(training_label_seq.shape)
    
    print(validation_label_seq[0])
    print(validation_label_seq[1])
    print(validation_label_seq[2])
    print(validation_label_seq.shape)


    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()


    num_epochs = 30
    history = model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)


    def plot_graphs(history, string):
      plt.plot(history.history[string])
      plt.plot(history.history['val_'+string])
      plt.xlabel("Epochs")
      plt.ylabel(string)
      plt.legend([string, 'val_'+string])
      plt.show()
      
    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")


    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    
    def decode_sentence(text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])


    e = model.layers[0]
    weights = e.get_weights()[0]
    print(weights.shape)
#tokenize_NLP_2()

### Takes IMDB reviews and rate them positive or nagative
def IMDB_NLP_1():
    # If the import fails, run this
    # !pip install -q tensorflow-datasets
    
    import tensorflow_datasets as tfds
    imdb, info = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)

    train_data, test_data = imdb['train'], imdb['test']

    tokenizer = info.features['text'].encoder

    print(tokenizer.subwords)

    sample_string = 'TensorFlow, from basics to mastery'
    
    tokenized_string = tokenizer.encode(sample_string)
    print ('Tokenized string is {}'.format(tokenized_string))
    
    original_string = tokenizer.decode(tokenized_string)
    print ('The original string: {}'.format(original_string))

    for ts in tokenized_string:
        print ('{} ----> {}'.format(ts, tokenizer.decode([ts])))

    
    BUFFER_SIZE = 10000
    BATCH_SIZE = 64
    
    train_dataset = train_data.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(train_dataset))
    test_dataset = test_data.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(test_data))


    embedding_dim = 64
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.summary()


    num_epochs = 10
    
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    history = model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)


    def plot_graphs(history, string):
      plt.plot(history.history[string])
      plt.plot(history.history['val_'+string])
      plt.xlabel("Epochs")
      plt.ylabel(string)
      plt.legend([string, 'val_'+string])
      plt.show()
      
    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")

    
    e = model.layers[0]
    weights = e.get_weights()[0]
    print(weights.shape) # shape: (vocab_size, embedding_dim)
#IMDB_NLP_1()

### Determines if sample text is sarcastic ###
def sarcasm_NLP_1():
    # https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json
    
    vocab_size = 10000
    embedding_dim = 16
    max_length = 100
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000


    with open(f"{os.getcwd()}/Data/sarcasm.json", 'r') as f:
        datastore = json.load(f)
    
    sentences = []
    labels = []
    
    for item in datastore:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])


    training_sentences = sentences[0:training_size]
    testing_sentences = sentences[training_size:]
    training_labels = labels[0:training_size]
    testing_labels = labels[training_size:]
    
    
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    
    word_index = tokenizer.word_index
    
    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    
    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    
    
    # Need this block to get it to work with TensorFlow 2.x
    training_padded = np.array(training_padded)
    training_labels = np.array(training_labels)
    testing_padded = np.array(testing_padded)
    testing_labels = np.array(testing_labels)
    
    
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    
    model.summary()
    
    
    num_epochs = 30
    history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)
    
    
    def plot_graphs(history, string):
      plt.plot(history.history[string])
      plt.plot(history.history['val_'+string])
      plt.xlabel("Epochs")
      plt.ylabel(string)
      plt.legend([string, 'val_'+string])
      plt.show()
      
    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")


    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    
    def decode_sentence(text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])
    
    print(decode_sentence(training_padded[0]))
    print(training_sentences[2])
    print(labels[2])
    
    
    e = model.layers[0]
    weights = e.get_weights()[0]
    print(weights.shape) # shape: (vocab_size, embedding_dim)
    
    
    sentence = ["granny starting to fear spiders in the garden might be real", "game of thrones season finale showing this sunday night"]
    sequences = tokenizer.texts_to_sequences(sentence)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    print(model.predict(padded))
#sarcasm_NLP_1()

### Overfitting with NLP ### not working bc a bad char in csv
def overfitting_NLP_1():
    # https://storage.googleapis.com/laurencemoroney-blog.appspot.com/training_cleaned.csv
    
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras import regularizers
    
    
    embedding_dim = 100
    max_length = 16
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size=160000
    test_portion=.1
    
    corpus = []
    
    # Note that I cleaned the Stanford dataset to remove LATIN1 encoding to make it easier for Python CSV reader
    # You can do that yourself with:
    # iconv -f LATIN1 -t UTF8 training.1600000.processed.noemoticon.csv -o training_cleaned.csv
    # I then hosted it on my site to make it easier to use in this notebook
    
    num_sentences = 0
    
    with open( f"{os.getcwd()}/Data/training_cleaned.csv", 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            list_item=[]
            list_item.append(row[5])
            this_label=row[0]
            if this_label=='0':
                list_item.append(0)
            else:
                list_item.append(1)
            num_sentences = num_sentences + 1
            corpus.append(list_item)

    print(num_sentences)
    print(len(corpus))
    print(corpus[1])
    
    
    sentences=[]
    labels=[]
    random.shuffle(corpus)
    for x in range(training_size):
        sentences.append(corpus[x][0])
        labels.append(corpus[x][1])
    
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    
    word_index = tokenizer.word_index
    vocab_size=len(word_index)
    
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    
    split = int(test_portion * training_size)
    
    test_sequences = padded[0:split]
    training_sequences = padded[split:training_size]
    test_labels = labels[0:split]
    training_labels = labels[split:training_size]
    
    
    print(vocab_size)
    print(word_index['i'])


    # Note this is the 100 dimension version of GloVe from Stanford
    # I unzipped and hosted it on my site to make this notebook easier
    # https://storage.googleapis.com/laurencemoroney-blog.appspot.com/glove.6B.100d.txt
    
    embeddings_index = {};
    with open(f'{os.getcwd()}/Data/glove.6B.100d.txt') as f:
        for line in f:
            values = line.split();
            word = values[0];
            coefs = np.asarray(values[1:], dtype='float32');
            embeddings_index[word] = coefs;
    
    embeddings_matrix = np.zeros((vocab_size+1, embedding_dim));
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word);
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector;

    print(len(embeddings_matrix))


    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length, weights=[embeddings_matrix], trainable=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()
    
    num_epochs = 50
    
    training_padded = np.array(training_sequences)
    training_labels = np.array(training_labels)
    testing_padded = np.array(test_sequences)
    testing_labels = np.array(test_labels)
    
    history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)
    
    print("Training Complete")
    
    
    import matplotlib.image  as mpimg
    import matplotlib.pyplot as plt
    
    #-----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    #-----------------------------------------------------------
    acc=history.history['accuracy']
    val_acc=history.history['val_accuracy']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    
    epochs=range(len(acc)) # Get number of epochs
    
    #------------------------------------------------
    # Plot training and validation accuracy per epoch
    #------------------------------------------------
    plt.plot(epochs, acc, 'r')
    plt.plot(epochs, val_acc, 'b')
    plt.title('Training and validation accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Accuracy", "Validation Accuracy"])
    
    plt.figure()
    
    #------------------------------------------------
    # Plot training and validation loss per epoch
    #------------------------------------------------
    plt.plot(epochs, loss, 'r')
    plt.plot(epochs, val_loss, 'b')
    plt.title('Training and validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Loss", "Validation Loss"])
    
    plt.figure()
#overfitting_NLP_1()

### shakespeare NLP ###
def shakespeare_NLP_1():
    # https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sonnets.txt
    
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras import regularizers
    import tensorflow.keras.utils as ku 

    tokenizer = Tokenizer()
    
    data = open(f'{os.getcwd()}/Data/sonnets.txt').read()
    
    corpus = data.lower().split("\n")
    
    
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    
    # create input sequences using list of tokens
    input_sequences = []
    for line in corpus:
    	token_list = tokenizer.texts_to_sequences([line])[0]
    	for i in range(1, len(token_list)):
    		n_gram_sequence = token_list[:i+1]
    		input_sequences.append(n_gram_sequence)
    
    
    # pad sequences 
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    
    # create predictors and label
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    
    label = ku.to_categorical(label, num_classes=total_words)


    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
    model.add(Bidirectional(LSTM(150, return_sequences = True)))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dense(total_words/2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    
    history = model.fit(predictors, label, epochs=100, verbose=1)

    
    acc = history.history['accuracy']
    loss = history.history['loss']
    
    epochs = range(len(acc))
    
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.title('Training accuracy')
    
    plt.figure()
    
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.title('Training loss')
    plt.legend()
    
    plt.show()
    
    
    seed_text = "Help me Obi Wan Kenobi, you're my only hope"
    next_words = 100
      
    for _ in range(next_words):
    	token_list = tokenizer.texts_to_sequences([seed_text])[0]
    	token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    	predicted = model.predict_classes(token_list, verbose=0)
    	output_word = ""
    	for word, index in tokenizer.word_index.items():
    		if index == predicted:
    			output_word = word
    			break
    	seed_text += " " + output_word
    print(seed_text)
#shakespeare_NLP_1()

### Creates and predicts seasonal data ###
def season_seq_1():
    from tensorflow import keras

    def plot_series(time, series, format="-", start=0, end=None):
        plt.plot(time[start:end], series[start:end], format)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.grid(True)
    
    def trend(time, slope=0):
        return slope * time
    
    def seasonal_pattern(season_time):
        """Just an arbitrary pattern, you can change it if you wish"""
        return np.where(season_time < 0.4,
                        np.cos(season_time * 2 * np.pi),
                        1 / np.exp(3 * season_time))
    
    def seasonality(time, period, amplitude=1, phase=0):
        """Repeats the same pattern at each period"""
        season_time = ((time + phase) % period) / period
        return amplitude * seasonal_pattern(season_time)
    
    def noise(time, noise_level=1, seed=None):
        rnd = np.random.RandomState(seed)
        return rnd.randn(len(time)) * noise_level
    
    time = np.arange(4 * 365 + 1, dtype="float32")
    baseline = 10
    series = trend(time, 0.1)  
    baseline = 10
    amplitude = 40
    slope = 0.05
    noise_level = 5
    
    # Create the series
    series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
    # Update with noise
    series += noise(time, noise_level, seed=42)
    
    plt.figure(figsize=(10, 6))
    plot_series(time, series)
    plt.show()


    split_time = 1000
    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]
    plt.figure(figsize=(10, 6))
    plot_series(time_train, x_train)
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plot_series(time_valid, x_valid)
    plt.show()
    
    
    naive_forecast = series[split_time - 1:-1]
    
    
    plt.figure(figsize=(10, 6))
    plot_series(time_valid, x_valid)
    plot_series(time_valid, naive_forecast)
    
    
    plt.figure(figsize=(10, 6))
    plot_series(time_valid, x_valid, start=0, end=150)
    plot_series(time_valid, naive_forecast, start=1, end=151)
    
    
    print(keras.metrics.mean_squared_error(x_valid, naive_forecast).numpy())
    print(keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy())
    
    
    def moving_average_forecast(series, window_size):
      """Forecasts the mean of the last few values.
         If window_size=1, then this is equivalent to naive forecast"""
      forecast = []
      for time in range(len(series) - window_size):
        forecast.append(series[time:time + window_size].mean())
      return np.array(forecast)
    
    
    moving_avg = moving_average_forecast(series, 30)[split_time - 30:]
    
    plt.figure(figsize=(10, 6))
    plot_series(time_valid, x_valid)
    plot_series(time_valid, moving_avg)
    
    
    print(keras.metrics.mean_squared_error(x_valid, moving_avg).numpy())
    print(keras.metrics.mean_absolute_error(x_valid, moving_avg).numpy())
    
    
    diff_series = (series[365:] - series[:-365])
    diff_time = time[365:]
    
    plt.figure(figsize=(10, 6))
    plot_series(diff_time, diff_series)
    plt.show()
    
    
    diff_moving_avg = moving_average_forecast(diff_series, 50)[split_time - 365 - 50:]
    
    plt.figure(figsize=(10, 6))
    plot_series(time_valid, diff_series[split_time - 365:])
    plot_series(time_valid, diff_moving_avg)
    plt.show()
    
    
    diff_moving_avg_plus_past = series[split_time - 365:-365] + diff_moving_avg
    
    plt.figure(figsize=(10, 6))
    plot_series(time_valid, x_valid)
    plot_series(time_valid, diff_moving_avg_plus_past)
    plt.show()
    
    
    print(keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_past).numpy())
    print(keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_past).numpy())
    
    
    diff_moving_avg_plus_smooth_past = moving_average_forecast(series[split_time - 370:-360], 10) + diff_moving_avg
    
    plt.figure(figsize=(10, 6))
    plot_series(time_valid, x_valid)
    plot_series(time_valid, diff_moving_avg_plus_smooth_past)
    plt.show()
    
    
    print(keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())
    print(keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())
#season_seq_1()

### creates and predicts seasonal data Predict with DNN###
def season_seq_2():
    def plot_series(time, series, format="-", start=0, end=None):
        plt.plot(time[start:end], series[start:end], format)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.grid(False)
    
    def trend(time, slope=0):
        return slope * time
    
    def seasonal_pattern(season_time):
        """Just an arbitrary pattern, you can change it if you wish"""
        return np.where(season_time < 0.1,
                        np.cos(season_time * 6 * np.pi),
                        2 / np.exp(9 * season_time))
    
    def seasonality(time, period, amplitude=1, phase=0):
        """Repeats the same pattern at each period"""
        season_time = ((time + phase) % period) / period
        return amplitude * seasonal_pattern(season_time)
    
    def noise(time, noise_level=1, seed=None):
        rnd = np.random.RandomState(seed)
        return rnd.randn(len(time)) * noise_level
    
    time = np.arange(10 * 365 + 1, dtype="float32")
    baseline = 10
    series = trend(time, 0.1)  
    baseline = 10
    amplitude = 40
    slope = 0.005
    noise_level = 3
    
    # Create the series
    series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
    # Update with noise
    series += noise(time, noise_level, seed=51)
    
    split_time = 3000
    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]
    
    window_size = 20
    batch_size = 32
    shuffle_buffer_size = 1000
    
    plot_series(time, series)
    
    
    def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
      dataset = tf.data.Dataset.from_tensor_slices(series)
      dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
      dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
      dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
      dataset = dataset.batch(batch_size).prefetch(1)
      return dataset
    
    
    dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
    
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(100, input_shape=[window_size], activation="relu"), 
        tf.keras.layers.Dense(10, activation="relu"), 
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9))
    model.fit(dataset,epochs=100,verbose=0)
    
    
    forecast = []
    for time in range(len(series) - window_size):
      forecast.append(model.predict(series[time:time + window_size][np.newaxis]))
    
    forecast = forecast[split_time-window_size:]
    results = np.array(forecast)[:, 0, 0]
    
    
    plt.figure(figsize=(10, 6))
    
    plot_series(time_valid, x_valid)
    plot_series(time_valid, results)
    
    
    tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()
#season_seq_2()

### creates and predicts seasonal data ###
def season_seq_3():
    def plot_series(time, series, format="-", start=0, end=None):
        plt.plot(time[start:end], series[start:end], format)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.grid(False)
    
    def trend(time, slope=0):
        return slope * time
    
    def seasonal_pattern(season_time):
        """Just an arbitrary pattern, you can change it if you wish"""
        return np.where(season_time < 0.1,
                        np.cos(season_time * 6 * np.pi),
                        2 / np.exp(9 * season_time))
    
    def seasonality(time, period, amplitude=1, phase=0):
        """Repeats the same pattern at each period"""
        season_time = ((time + phase) % period) / period
        return amplitude * seasonal_pattern(season_time)
    
    def noise(time, noise_level=1, seed=None):
        rnd = np.random.RandomState(seed)
        return rnd.randn(len(time)) * noise_level
    
    time = np.arange(10 * 365 + 1, dtype="float32")
    baseline = 10
    series = trend(time, 0.1)  
    baseline = 10
    amplitude = 40
    slope = 0.005
    noise_level = 3
    
    # Create the series
    series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
    # Update with noise
    series += noise(time, noise_level, seed=51)
    
    split_time = 3000
    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]
    
    window_size = 20
    batch_size = 32
    shuffle_buffer_size = 1000
    
    plot_series(time, series)
    
    
    def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
      dataset = tf.data.Dataset.from_tensor_slices(series)
      dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
      dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
      dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
      dataset = dataset.batch(batch_size).prefetch(1)
      return dataset
    
    
    tf.keras.backend.clear_session()
    tf.random.set_seed(51)
    np.random.seed(51)
    
    tf.keras.backend.clear_session()
    dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
    
    model = tf.keras.models.Sequential([
      tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                          input_shape=[None]),
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
      tf.keras.layers.Dense(1),
      tf.keras.layers.Lambda(lambda x: x * 10.0)
    ])
    
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-8 * 10**(epoch / 20))
    optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])
    history = model.fit(dataset, epochs=100, callbacks=[lr_schedule])
    
    
    plt.semilogx(history.history["lr"], history.history["loss"])
    plt.axis([1e-8, 1e-4, 0, 30])
    
    
    tf.keras.backend.clear_session()
    tf.random.set_seed(51)
    np.random.seed(51)
    
    tf.keras.backend.clear_session()
    dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
    
    model = tf.keras.models.Sequential([
      tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                          input_shape=[None]),
       tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
      tf.keras.layers.Dense(1),
      tf.keras.layers.Lambda(lambda x: x * 100.0)
    ])
    
    
    model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9),metrics=["mae"])
    history = model.fit(dataset,epochs=100,verbose=1)
    
    
    forecast = []
    results = []
    for time in range(len(series) - window_size):
      forecast.append(model.predict(series[time:time + window_size][np.newaxis]))
    
    forecast = forecast[split_time-window_size:]
    results = np.array(forecast)[:, 0, 0]
    
    
    plt.figure(figsize=(10, 6))
    
    plot_series(time_valid, x_valid)
    plot_series(time_valid, results)
    
    
    tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()
    

    #-----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    #-----------------------------------------------------------
    mae=history.history['mae']
    loss=history.history['loss']
    
    epochs=range(len(loss)) # Get number of epochs
    
    #------------------------------------------------
    # Plot MAE and Loss
    #------------------------------------------------
    plt.plot(epochs, mae, 'r')
    plt.plot(epochs, loss, 'b')
    plt.title('MAE and Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["MAE", "Loss"])
    
    plt.figure()
    
    epochs_zoom = epochs[200:]
    mae_zoom = mae[200:]
    loss_zoom = loss[200:]
    
    #------------------------------------------------
    # Plot Zoomed MAE and Loss
    #------------------------------------------------
    plt.plot(epochs_zoom, mae_zoom, 'r')
    plt.plot(epochs_zoom, loss_zoom, 'b')
    plt.title('MAE and Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["MAE", "Loss"])
    
    plt.figure()
#season_seq_3()

### uses LSTMs / DNN to predict historic temp values ###
def temp_seq_1():
    # https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv

    def plot_series(time, series, format="-", start=0, end=None):
        plt.plot(time[start:end], series[start:end], format)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.grid(True)
    
    
    time_step = []
    temps = []
    
    with open(f'{os.getcwd()}/Data/daily-min-temperatures.csv') as csvfile:
      reader = csv.reader(csvfile, delimiter=',')
      next(reader)
      step=0
      for row in reader:
        temps.append(float(row[1]))
        time_step.append(step)
        step = step + 1
    
    series = np.array(temps)
    time = np.array(time_step)
    plt.figure(figsize=(10, 6))
    plot_series(time, series)
    
    
    split_time = 2500
    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]
    
    window_size = 30
    batch_size = 32
    shuffle_buffer_size = 1000
    
    
    def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
        series = tf.expand_dims(series, axis=-1)
        ds = tf.data.Dataset.from_tensor_slices(series)
        ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(window_size + 1))
        ds = ds.shuffle(shuffle_buffer)
        ds = ds.map(lambda w: (w[:-1], w[1:]))
        return ds.batch(batch_size).prefetch(1)
    
    
    def model_forecast(model, series, window_size):
        ds = tf.data.Dataset.from_tensor_slices(series)
        ds = ds.window(window_size, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(window_size))
        ds = ds.batch(32).prefetch(1)
        forecast = model.predict(ds)
        return forecast
    
    
    tf.keras.backend.clear_session()
    tf.random.set_seed(51)
    np.random.seed(51)
    window_size = 64
    batch_size = 256
    train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
    print(train_set)
    print(x_train.shape)
    
    model = tf.keras.models.Sequential([
      tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                          strides=1, padding="causal",
                          activation="relu",
                          input_shape=[None, 1]),
      tf.keras.layers.LSTM(64, return_sequences=True),
      tf.keras.layers.LSTM(64, return_sequences=True),
      tf.keras.layers.Dense(30, activation="relu"),
      tf.keras.layers.Dense(10, activation="relu"),
      tf.keras.layers.Dense(1),
      tf.keras.layers.Lambda(lambda x: x * 400)
    ])
    
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-8 * 10**(epoch / 20))
    optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])
    history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])
    
    
    plt.semilogx(history.history["lr"], history.history["loss"])
    plt.axis([1e-8, 1e-4, 0, 60])
    
    
    tf.keras.backend.clear_session()
    tf.random.set_seed(51)
    np.random.seed(51)
    train_set = windowed_dataset(x_train, window_size=60, batch_size=100, shuffle_buffer=shuffle_buffer_size)
    model = tf.keras.models.Sequential([
      tf.keras.layers.Conv1D(filters=60, kernel_size=5,
                          strides=1, padding="causal",
                          activation="relu",
                          input_shape=[None, 1]),
      tf.keras.layers.LSTM(60, return_sequences=True),
      tf.keras.layers.LSTM(60, return_sequences=True),
      tf.keras.layers.Dense(30, activation="relu"),
      tf.keras.layers.Dense(10, activation="relu"),
      tf.keras.layers.Dense(1),
      tf.keras.layers.Lambda(lambda x: x * 400)
    ])
    
    
    optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])
    history = model.fit(train_set,epochs=150)
    
    
    rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
    rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]
    
    
    plt.figure(figsize=(10, 6))
    plot_series(time_valid, x_valid)
    plot_series(time_valid, rnn_forecast)
    
    tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()
    
    
    print(rnn_forecast)
#temp_seq_1()

### Predicts sunspot occurences ###
def sunspot_seq_1():
    # https://storage.googleapis.com/laurencemoroney-blog.appspot.com/Sunspots.csv

    def plot_series(time, series, format="-", start=0, end=None):
        plt.plot(time[start:end], series[start:end], format)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.grid(True)
    
    time_step = []
    sunspots = []
    
    with open(f'{os.getcwd()}/Data/sunspots.csv') as csvfile:
      reader = csv.reader(csvfile, delimiter=',')
      next(reader)
      for row in reader:
        sunspots.append(float(row[2]))
        time_step.append(int(row[0]))
    
    series = np.array(sunspots)
    time = np.array(time_step)
    plt.figure(figsize=(10, 6))
    plot_series(time, series)    
        
        
    series = np.array(sunspots)
    time = np.array(time_step)
    plt.figure(figsize=(10, 6))
    plot_series(time, series)    
        
        
    split_time = 3000
    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]
    
    window_size = 30
    batch_size = 32
    shuffle_buffer_size = 1000    
        
        
    def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
        series = tf.expand_dims(series, axis=-1)
        ds = tf.data.Dataset.from_tensor_slices(series)
        ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(window_size + 1))
        ds = ds.shuffle(shuffle_buffer)
        ds = ds.map(lambda w: (w[:-1], w[1:]))
        return ds.batch(batch_size).prefetch(1)    
        
    
    def model_forecast(model, series, window_size):
        ds = tf.data.Dataset.from_tensor_slices(series)
        ds = ds.window(window_size, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(window_size))
        ds = ds.batch(32).prefetch(1)
        forecast = model.predict(ds)
        return forecast    
        
        
    tf.keras.backend.clear_session()
    tf.random.set_seed(51)
    np.random.seed(51)
    window_size = 64
    batch_size = 256
    train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
    print(train_set)
    print(x_train.shape)
    
    model = tf.keras.models.Sequential([
      tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                          strides=1, padding="causal",
                          activation="relu",
                          input_shape=[None, 1]),
      tf.keras.layers.LSTM(64, return_sequences=True),
      tf.keras.layers.LSTM(64, return_sequences=True),
      tf.keras.layers.Dense(30, activation="relu"),
      tf.keras.layers.Dense(10, activation="relu"),
      tf.keras.layers.Dense(1),
      tf.keras.layers.Lambda(lambda x: x * 400)
    ])
    
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-8 * 10**(epoch / 20))
    optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])
    history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])
        
        
    plt.semilogx(history.history["lr"], history.history["loss"])
    plt.axis([1e-8, 1e-4, 0, 60])    
        
        
    tf.keras.backend.clear_session()
    tf.random.set_seed(51)
    np.random.seed(51)
    train_set = windowed_dataset(x_train, window_size=60, batch_size=100, shuffle_buffer=shuffle_buffer_size)
    model = tf.keras.models.Sequential([
      tf.keras.layers.Conv1D(filters=60, kernel_size=5,
                          strides=1, padding="causal",
                          activation="relu",
                          input_shape=[None, 1]),
      tf.keras.layers.LSTM(60, return_sequences=True),
      tf.keras.layers.LSTM(60, return_sequences=True),
      tf.keras.layers.Dense(30, activation="relu"),
      tf.keras.layers.Dense(10, activation="relu"),
      tf.keras.layers.Dense(1),
      tf.keras.layers.Lambda(lambda x: x * 400)
    ])
    
    
    optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])
    history = model.fit(train_set,epochs=500)   
        
        
    rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
    rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]    
        
        
    plt.figure(figsize=(10, 6))
    plot_series(time_valid, x_valid)
    plot_series(time_valid, rnn_forecast)    
        
        
    tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()
    
    
    #-----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    #-----------------------------------------------------------
    loss=history.history['loss']
    
    epochs=range(len(loss)) # Get number of epochs
    
    
    #------------------------------------------------
    # Plot training and validation loss per epoch
    #------------------------------------------------
    plt.plot(epochs, loss, 'r')
    plt.title('Training loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Loss"])
    
    plt.figure()
    
    
    
    zoomed_loss = loss[200:]
    zoomed_epochs = range(200,500)
    
    
    #------------------------------------------------
    # Plot training and validation loss per epoch
    #------------------------------------------------
    plt.plot(zoomed_epochs, zoomed_loss, 'r')
    plt.title('Training loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Loss"])
    
    plt.figure()
    
    
    print(rnn_forecast)
#sunspot_seq_1()

### Predicts sunspot occurences using a DNN ###
def sunspot_seq_2():
    # https://storage.googleapis.com/laurencemoroney-blog.appspot.com/Sunspots.csv

    def plot_series(time, series, format="-", start=0, end=None):
        plt.plot(time[start:end], series[start:end], format)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.grid(True)
    
    time_step = []
    sunspots = []
    
    with open(f'{os.getcwd()}/Data/sunspots.csv') as csvfile:
      reader = csv.reader(csvfile, delimiter=',')
      next(reader)
      for row in reader:
        sunspots.append(float(row[2]))
        time_step.append(int(row[0]))
    
    series = np.array(sunspots)
    time = np.array(time_step)
    plt.figure(figsize=(10, 6))
    plot_series(time, series)    
        
        
    series = np.array(sunspots)
    time = np.array(time_step)
    plt.figure(figsize=(10, 6))
    plot_series(time, series)    
        
        
    split_time = 3000
    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]
    
    window_size = 30
    batch_size = 32
    shuffle_buffer_size = 1000    
        
        
    def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
        series = tf.expand_dims(series, axis=-1)
        ds = tf.data.Dataset.from_tensor_slices(series)
        ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(window_size + 1))
        ds = ds.shuffle(shuffle_buffer)
        ds = ds.map(lambda w: (w[:-1], w[1:]))
        return ds.batch(batch_size).prefetch(1)    

    dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(20, input_shape=[window_size], activation="relu"), 
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-7, momentum=0.9))
    model.fit(dataset,epochs=50,verbose=0)
    
    
    forecast=[]
    for time in range(len(series) - window_size):
      forecast.append(model.predict(series[time:time + window_size][np.newaxis]))
    
    forecast = forecast[split_time-window_size:]
    results = np.array(forecast)[:, 0, 0]
    
    
    plt.figure(figsize=(10, 6))
    
    plot_series(time_valid, x_valid)
    plot_series(time_valid, results)
    
    
    tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()
sunspot_seq_2()




































































