---

# 📚 **TextCNN: PyTorch Text Classification**

This repository contains a simple implementation of a **Convolutional Neural Network (CNN)** for text classification using PyTorch. The model is designed to classify text into predefined categories, such as sentiment analysis or topic classification.

---

## 🚀 **Features**

- **TextCNN Model**: A lightweight CNN architecture for text classification.
- **Custom Dataset**: Supports loading and preprocessing text data.
- **Training Pipeline**: Includes training, validation, and evaluation loops.
- **GPU Support**: Automatically uses GPU if available.
- **Modular Design**: Easy to extend and customize.

---

## 📋 **Requirements**

To run this project, you need the following dependencies:

- Python 3.7+
- PyTorch 1.8+
- NumPy
- torchtext (optional, for data preprocessing)

Install the required packages using:

```bash
pip install -r requirements.txt
```

---

## 📂 **Project Structure**

```
.
├── data/                    # Folder for storing datasets
├── models/                  # Model definitions
│   └── text_cnn.py          # TextCNN model implementation
├── utils/                   # Utility functions
│   └── dataset.py           # Custom dataset class
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## 🔧 **How to Use**

### 1. **Prepare Your Dataset**

- Place your text data in the `data/` folder.
- Ensure your dataset is split into training and validation sets.
- Preprocess the text data (e.g., tokenization, lowercasing) and convert it into a format compatible with the `TextDataset` class.

### 2. **Train the Model**

Run the training script to train the TextCNN model:

```bash
python train.py --data_path data/train.csv --epochs 10 --batch_size 64
```

### 3. **Evaluate the Model**

After training, evaluate the model on the validation set:

```bash
python evaluate.py --model_path models/textcnn.pth --data_path data/val.csv
```

### 4. **Customize the Model**

You can modify the `models/text_cnn.py` file to experiment with different architectures, such as:

- Adding more convolutional layers.
- Changing the filter sizes or number of filters.
- Adjusting the dropout rate.

---

## 📊 **Results**

Here are some example results from training the TextCNN model on a sentiment analysis dataset:

| Epoch | Training Loss | Validation Loss | Accuracy |
|-------|---------------|------------------|----------|
| 1     | 0.693         | 0.680            | 55.2%    |
| 5     | 0.450         | 0.420            | 81.5%    |
| 10    | 0.200         | 0.350            | 88.7%    |

---

## 📄 **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙌 **Contributing**

Contributions are welcome! If you have any ideas, suggestions, or bug fixes, feel free to open an issue or submit a pull request.

---

## 📧 **Contact**

If you have any questions or need further assistance, feel free to contact me at [snv123abc@hotmail.com](mailto:snv123abc@hotmail.com).

---

## 🌟 **Acknowledgments**

- Inspired by the [TextCNN paper](https://arxiv.org/abs/1408.5882).
- Special thanks to the PyTorch community for their excellent documentation and tutorials.

---

Feel free to customize this `README.md` to better fit your project!
