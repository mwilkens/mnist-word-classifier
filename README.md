# MNIST-Trained Word Classifier

Reads words and letters from images and outputs them
Trained by the MNIST Letter database

## Tasks List

1. Prep sample images ( with handwritten words ) (should have some constructed with MNIST letters)
2. **[DONE]** Create some way of sharing our code with each other (github probably)
3. Create framework algorithm with dummy functions as classifiers
    1. **[DONE]** parse the MNIST files
    2. Classifier training implimentation
    3. Read image from file (_what format?_)
    4. Convolve through image
    5. Mark centroids and letters where we find them
    6. construct word and output string
4. Create letter classifier
5. Create MNIST classifier
6. Create Word construction algorithm
7. Create presentation!!!

## MNIST Classifier

Training Function

    (inputs: training set, testing set)

    (outputs: model, loss)

Predicting Function

    (inputs: 20x20 image)

    (outputs: letter, loss)

## Letter Classifier

Training Function

    (inputs: training set*, testing set*)

    (outputs: model, loss)

Predicting Function

    (inputs: 50x50 image)

    (outputs: prediction probability)

\* MNIST database? maybe with some random blanks?

Use MSER Feature Detection?
* https://www.mathworks.com/help/vision/examples/automatically-detect-and-recognize-text-in-natural-images.html
* (This would deviate from our algorithm)

## Word Constructor

Constructor Function

    (inputs: letter array, centroids array)

    (outputs: string)


## Algorithm

1. Load in image with handwriting
2. Convolve N by N box around the image
  1. for each box, use a simple letter classifier to determine whether certain pixel areas are likely to be letters.
  2. if we've reached a peak, identify the centroid and move on (making sure we reduce overlap somehow)
3. When we're done convolving, take the centroids, extract letter information and classify each letter using our MNIST classifier
4. construct a word based on the output of the MNIST classifier but also from the centroid distance (for spaces)
