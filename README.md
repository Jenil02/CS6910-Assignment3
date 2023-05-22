# CS6910-Assignment3
# CS6910 - Fundamentals of Deep Learning - Assignment 3
### Author : Jenil Sheth (BE20B031)

This repository contains the code for the third assignment of the course CS6910 - Fundamentals of Deep Learning. In this assignment, I have built a RNN/LSTM/GRU based seq2seq model using pytorch.

## Code structure

`CS6910 - Assignment3` folder contains the following files:
- `Assignment_3.ipynb` : This file contains the seq2seq model with attention.
- `fdl-a3.ipynb` : This file contains the seq2seq model without attention.
- `predictions_vanilla.txt` : This file contains the predictions on test set from the vanilla seq2seq model 

- `predictions_attention.txt` : This file contains the predictions on test set from the attention seq2seq model 

## Running the model
You can run the model with few commands as given below

train_pairs, val_pairs, test_pairs, input_lang, output_lang = readLang(input_language, output_language)

encoder = Encoder(model_type, input_lang.n_letters, embedding_size, hidden_size, dropout, no_of_hidden_layers).to(device)
decoder = AttnDecoder(model_type, output_lang.n_letters, hidden_size, dropout, no_of_hidden_layers).to(device)

train = Train(train_pairs, encoder, decoder, nn.NLLLoss(), teacher_forcing_ratio)
train.trainIters(optimizer, learning_rate, print_every=1000, epochs)

test_acc = train.evaluateData(test_pairs)

See the .ipynb files to get a deeper understanding of the model
