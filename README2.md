# Part 2: Implementation
The dataset which I used to train the model is taken from kaggle.
- It's link is (https://www.openai.com](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)
- The dataset is published in four versions: for-original, for-norm, for-2sec and for-rerec.
- The first version, named for-original, contains the files as collected from the speech sources, without any modification (balanced version).
- The second version, called for-norm, contains the same files, but balanced in terms of gender and class and normalized in terms of sample rate, volume and number of channels.
- The third one, named for-2sec is based on the second one, but with the files truncated at 2 seconds.
- The last version, named for-rerec, is a rerecorded version of the for-2second dataset, to simulate a scenario where an attacker sends an utterance through a voice channel (i.e. a phone call or a voice message).

  I chose this dataset because it contains variety of voices i.e., recorded and normal of upto 2 seconds.

  ## Steps to run the code
  ### Step 1: Setup Environment
  - Install required packages:
    ``` 
    pip install torch torchaudio librosa tqdm scikit-learn numpy
    ```
 ### Step 2: Save the Full Code
 Save the complete code you provided as app.py in your project folder.
 ### Step 3: Run Different Modes
  #### A. Train the Model
  ```
  python app.py train
  ```
- Trains on all datasets (for-2sec, for-norm, etc.)
- Saves best model as crnn_unified_model.pth
- Expected time: 30 minutes - several hours (depending on GPU)
  
  #### B. Evaluate on Test Data
  ```
  python app.py test
  ```
- Loads saved model
- Computes accuracy on all test sets
- Expected time: 5-20 minutes

#### C. Predict Single Audio File
  ```
  python app.py predict path/to/your/audio.wav
  ```
- Outputs: REAL or FAKE with confidence score
- Expected time: <1 second

#### Now for calculating the performance matrix
```
python evaluate_metrics.py
```
