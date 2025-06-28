# makemore_passwords

This project leverages a character-level language model, inspired by Andrej Karpathy's makemore series, to generate new "passwords" after being trained on a dataset of leaked Indian passwords. The goal is to explore the patterns within these passwords and generate similar-looking sequences, which can be useful for understanding common password structures or for security research.

## Features

- **Customizable Language Models**: Supports various language model architectures including:
  - Bigram
  - MLP (Multi-Layer Perceptron)
  - RNN (Recurrent Neural Network)
  - GRU (Gated Recurrent Unit)
  - Bag of Words (BoW)
  - Transformer (a simplified version of GPT-2)

- **Training on Custom Datasets**: Easily train the model on any text file with one item per line (e.g., a list of passwords, names, etc.).

- **Evaluation Metrics**: Tracks training and testing loss to monitor model performance.

- **Sample Generation**: Generates new sequences based on the trained model.

- **Configurable Training Parameters**: Control batch size, learning rate, number of layers, embedding dimensions, and more.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You'll need Python 3 and PyTorch installed.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install tensorboard
```

### Installation

Clone the repository (assuming `makemore.py` is part of a larger repository):

```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

If `makemore.py` is a standalone script, just save it to your desired directory.

Place your `passwords.txt` file (or any other text file you wish to train on, with one password per line) in the same directory as `makemore.py`.

## Usage

### Training a Model

To train a new model to generate passwords, run the `makemore.py` script with your input file.

```bash
python3 makemore.py -i passwords.txt -o out_passwords --type transformer
```

- `-i passwords.txt`: Specifies your input file containing passwords (one per line).
- `-o out_passwords`: Sets the output directory for the trained model and logs.
- `--type transformer`: Chooses the model architecture (`transformer`, `gru`, `rnn`, `mlp`, `bow`, `bigram`).

You can also adjust other parameters:

```bash
python3 makemore.py \
  -i passwords.txt \
  -o out_passwords_gru \
  --type gru \
  --batch-size 64 \
  --learning-rate 1e-3 \
  --n-layer 6 \
  --n-embd 128 \
  --max-steps 5000
```

### Resuming Training

If you have a previously trained model in your work-dir (e.g., `out_passwords`), you can resume training from its last saved state:

```bash
python3 makemore.py -i passwords.txt -o out_passwords --resume --type transformer
```

> **Note**: Make sure to specify the same `--type` as the model you are resuming.

### Generating Passwords

To generate new passwords using a trained model without further training, use the `--sample-only` flag:

```bash
python3 makemore.py -i passwords.txt -o out_passwords --type transformer --sample-only
```

This will load the model from `out_passwords/model.pt` and print 50 generated samples to the console, categorized by whether they appeared in the training set, test set, or are entirely new.

## Model Types

You can specify the model type using the `--type` argument:

- `bigram`: A simple model that predicts the next character based on the current one.
- `mlp`: A multi-layer perceptron.
- `rnn`: A basic Recurrent Neural Network.
- `gru`: A Gated Recurrent Unit, a more advanced RNN.
- `bow`: Bag of Words model with a causal attention mechanism.
- `transformer`: A simplified Transformer architecture, similar to GPT-2.

## Project Structure

```
.
├── makemore.py        # The main script for training and generating
├── passwords.txt      # Your input dataset of passwords (one per line)
└── out/               # Default output directory for models and logs
    ├── model.pt           # Saved model weights
    └── events.out.tfevents # TensorBoard logs
```

## Contributing

Feel free to fork the repository and submit pull requests.

## License

This project is open-source. (Consider adding a specific license, e.g., MIT, Apache 2.0)

## Acknowledgments

- Inspired by Andrej Karpathy's [makemore](https://github.com/karpathy/makemore) series.
- The core Transformer code is a simplified version of [minGPT](https://github.com/karpathy/minGPT).