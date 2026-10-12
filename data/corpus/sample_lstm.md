# Long Short-Term Memory (LSTM)

Long Short-Term Memory networks are a type of recurrent neural network designed to capture long-range dependencies in sequential data.

The core idea is that standard RNNs struggle with vanishing gradients during backpropagation through time. LSTMs address this by introducing gates that regulate how information enters, leaves, and persists in the cell state.

## Cell state and gates

An LSTM cell contains a cell state and three main gates:

- The forget gate decides what information to discard from the previous cell state.
- The input gate decides what new information should be stored.
- The output gate decides what information should be passed to the next hidden state.

The gates use sigmoid and tanh activations to control the flow of information. This allows the network to retain relevant information over long sequences while forgetting irrelevant details.

## Why LSTMs matter

LSTMs are especially useful for time series, speech, language modeling, and other tasks where context from earlier steps matters. Compared with vanilla RNNs, they are more stable during training and better at learning dependencies over longer intervals.

## Interview-style explanation

An LSTM is often described as a memory cell with selective write, read, and forget behavior. The hidden state and cell state are updated using learned parameters, which let the model decide whether to keep, update, or remove information.

## Key takeaway

LSTMs solve the vanishing gradient problem by giving the model explicit mechanisms for controlling memory over time. They remain a foundational architecture in sequence modeling.
