# ProjCVex4
Repo for face detection and recognition for ProjectCV exercise 4. SoSe23

Summary and Action Points for the Assignment:

Assignment Goal: Develop a basic face recognition system with training and testing modules.

Action Points:

1. Exercise 4.1: Face Detection, Tracking, and Alignment
   - Implement the track_face method for face tracking using template matching.
   - Perform face detection in the first frame using the detect_face method.
   - Use template matching to track the face in subsequent frames.
   - Re-initialize the tracker using MTCNN if the track is lost.
   - Integrate alignment for pose normalization.

2. Exercise 4.2: Face Identification and Verification
   - Integrate the FaceNet class for feature extraction in the FaceRecognizer.
   - Implement the update method to store training samples in the gallery.
   - Implement the predict method for face identification using k-NN.
   - Extend the predict method to return posterior probabilities and distances.
   - Test face identification with different persons and adjust k-NN parameters.
   - Challenge the algorithm's robustness with profile views, expression variations, and illumination changes.
   - Implement open-set identification using distance thresholds and posterior probabilities.
   - Test open-set identification with unknown persons and determine suitable thresholds.

3. Exercise 4.3: Face Clustering
   - Implement the update method to store embeddings for new faces.
   - Implement the k-means algorithm and fit method for clustering.
   - Analyze the convergence behavior and sensitivity of the clustering.
   - Implement the predict method for re-identification by finding the best matching cluster.
   - Test re-identification with facial data from multiple persons.

4. Exercise 4.4: Evaluation of Face Recognition
   - Complete the evaluation by implementing the run method.
   - Implement the calc_identification_rate method to compute the identification rate at rank 1.
   - Implement the select_similarity_threshold to determine thresholds for false alarm rates.
   - Fit the classifier on the training data and predict similarities on the test data.
   - Compute the identification rates for different false alarm rates.
   - Use the DIR curve to select suitable similarity thresholds based on the requirements.

Priority Tasks:

1. Exercise 4.1: Implement face tracking and alignment.
2. Exercise 4.2: Integrate FaceNet for feature extraction and implement face identification.
3. Exercise 4.3: Implement face clustering and re-identification.
4. Exercise 4.4: Complete the evaluation by implementing the run method.

It's important to follow the assignment instructions, review the provided skeleton code, and refer to any additional hints or details mentioned in the documentation. Adjust parameters, thresholds, and algorithms based on testing and evaluation results to optimize the performance of the face recognition system.
