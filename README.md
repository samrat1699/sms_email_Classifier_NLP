# Spam Classifier for SMS and Email

## Description

This project is a Streamlit web application that classifies SMS and email messages as spam or ham (not spam). It utilizes natural language processing (NLP) techniques and machine learning models to analyze and predict the probability of a message being spam. The application supports vectorizing text using TF-IDF and predicting using a trained Multinomial Naive Bayes model.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Application](#Application)

## Installation

To set up and run this project locally, follow these steps:

1. **Clone the repository:**

    ```sh
    git clone https://github.com/your-username/spam-classifier.git
    cd spam-classifier
    ```

2. **Create a virtual environment:**

    ```sh
    python -m venv env
    ```

3. **Activate the virtual environment:**

    - On Windows:

        ```sh
        .\env\Scripts\activate
        ```

    - On macOS and Linux:

        ```sh
        source env/bin/activate
        ```

4. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

5. **Download the spaCy language model:**

    ```sh
    python -m spacy download en_core_web_sm
    ```

6. **Ensure you have the necessary model files:**

    - `Smsvectorizer.pkl`
    - `Smsmodel.pkl`
    - `Emailvectorizer.pkl`
    - `emailmodel.pkl`

    These files should be in the same directory as your script or specify the correct paths in the code.

## Usage

To run the Streamlit app, execute the following command in your terminal:

```sh
streamlit run smsemail.py
 ```
## Application 
You can also access the deployed application online:
 ```sh
[Spam classifier for sms and email](https://smsemailclassifiernlp.streamlit.app/).
 ```



