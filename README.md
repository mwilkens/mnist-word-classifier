# MNIST-Trained Word Classifier

Reads words and letters from images and outputs them
Trained by the MNIST Letter database

## MNIST Classifier

Training Function

    (inputs: training set, testing set)

    (outputs: model, loss)

Predicting Function

    (inputs: NxN* image)

    (outputs: letter, loss)

\* what is N

## Letter Classifier

Training Function

    (inputs: training set*, testing set*)

    (outputs: model, loss)

Predicting Function

    (inputs: 50x50 image)

    (outputs: prediction probability)

\* MNIST database? maybe with some random blanks?

## Algorithm

1. Load in image with handwriting
2. Convolve N by N box around the image
  1. for each box, use a simple letter classifier to determine whether certain pixel areas are likely to be letters.
  2. if we've reached a peak, identify the centroid and move on (making sure we reduce overlap somehow)
3. When we're done convolving, take the centroids, extract letter information and classify each letter using our MNIST classifier
4. construct a word based on the output of the MNIST classifier but also from the centroid distance (for spaces)
