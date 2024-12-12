Objectives:
Develop a sentiment analysis model using Long Short-Term Memory (LSTM) networks.
Preprocess text data to ensure model compatibility and handle sequence variations.
Achieve high accuracy in binary sentiment classification.
Provide a reusable and scalable model for future applications.
Methodology:
Dataset:

Used a labeled dataset (amazon_cells_labelled.txt) containing reviews and corresponding sentiment labels.
Data Preprocessing:

Identified and logged unusual characters in reviews for cleaning.
Tokenized the text data into sequences using a Tokenizer.
Padded sequences to a fixed length (95th percentile of sequence lengths) for consistency.
Model Architecture:

LSTM layers for sequential data processing:
First LSTM: 128 units with return sequences enabled.
Second LSTM: 64 units for deeper feature extraction.
Dropout layers to prevent overfitting.
Dense layers for feature fusion and binary classification with sigmoid activation.
Training:

Optimizer: Adam.
Loss Function: Binary crossentropy.
Early stopping to halt training when validation loss stopped improving.
Evaluation:

Used metrics like accuracy and the classification report.
Visualized training and validation loss/accuracy across epochs.
The test set confirmed the model's performance on unseen data.
Model Deployment:

Saved the trained model as sentiment_analysis_model.h5 for reuse in inference or further training.
Outcomes:
The model achieved strong performance metrics, indicating its suitability for binary sentiment analysis.
Training and validation metrics suggest balanced learning without significant overfitting or underfitting.
Key Improvements and Next Steps:
Expand the dataset to include diverse and nuanced sentiments.
Experiment with hyperparameter tuning and advanced architectures like bidirectional LSTMs or GRUs.
Test on real-world, noisy data to assess generalization capabilities.
This project provides a foundation for automating sentiment analysis tasks, beneficial in areas like customer service and marketing analytics.
