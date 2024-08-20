# AutoML Streamlit App

This repository contains a Streamlit application that allows users to train, evaluate, and apply machine learning models using PyCaret. The app supports both classification and regression tasks.

## Features

- **Model Training**: Train machine learning models using PyCaret's autoML functionality.
- **Model Evaluation**: View performance metrics of the trained models.
- **Model Inference**: Apply the trained model to new data for predictions. Supports both single predictions and batch predictions.
- **Download Predictions**: Save the predictions to a CSV file for further analysis.

## Installation

### Prerequisites

- Python 3.7+
- Anaconda (recommended for managing environments)
- Git

### Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/AutoML_Streamlit_App_POC.git
   cd AutoML_Streamlit_App_POC
   ```

2. **Create a virtual environment**:

   ```bash
   conda create -n tokyo python=3.8
   conda activate tokyo
   ```

3. **Install the dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

## Usage

### 1. **Home Page**
   - Start by selecting the type of task you want to perform: Classification or Regression.
   - Upload your dataset, and the app will automatically train models using PyCaret.

### 2. **Model Evaluation**
   - View the performance metrics of the trained models.
   - Select the best model based on the evaluation metrics.

### 3. **Apply Model**
   - Apply the trained model to new data.
   - Choose between single prediction and batch prediction.
   - Download the predictions as a CSV file.

### 4. **Custom Settings**
   - Adjust settings like model hyperparameters, preprocessing options, and more.

## Project Structure

- `app.py`: Main entry point for the Streamlit app.
- `pages/`: Contains separate pages for applying the model and model evaluation.
- `requirements.txt`: List of Python packages required to run the app.
- `README.md`: Documentation for the project.

## Troubleshooting

### Common Issues

1. **NaN or Inf Values**: Ensure that your data doesn't contain NaN or infinite values before applying the model. The app provides handling for such values but might need adjustments based on your dataset.
2. **Duplicate Columns**: The app automatically handles duplicate columns that may arise during the prediction phase.

### Debugging

- Check the Streamlit app logs for any errors. Logs are usually displayed in the terminal where you ran the app.

## Contributing

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/yourFeature`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/yourFeature`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

