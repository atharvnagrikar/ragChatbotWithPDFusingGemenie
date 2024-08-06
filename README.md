# Chat with PDF using Gemini

This project allows you to interact with PDF documents using a Streamlit app. The app leverages Google’s Generative AI to answer questions based on the content of the uploaded PDFs.

## Features

- **PDF Upload**: Upload multiple PDF documents for processing.
- **Question Answering**: Ask questions related to the content of the uploaded PDFs.
- **Generative AI Integration**: Utilizes Google’s Gemini Generative AI for answering questions.

## Requirements

To run this application, you'll need:

- **Google API Key**: You'll need a Google API key to access Google’s Generative AI services.
- **Python 3.10.14**: Ensure you have Python 3.7 or later installed.

## Installation

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/your-username/your-repository.git
    cd your-repository
    ```

2. **Create and Activate a Virtual Environment**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Streamlit App**:

    ```bash
    streamlit run app.py
    ```

2. **Open the Application**:

    - Open your browser and navigate to `http://localhost:8501`.

3. **Using the App**:

    - **Upload PDFs**: Go to the sidebar, use the "Upload your PDF Files" button to upload your PDF documents.
    - **Enter Google API Key**: Input your Google API key in the sidebar.
    - **Ask a Question**: Enter your question in the provided text input field.
    - **Submit**: Click on the "Submit & Process" button to process the PDFs and create embeddings.
    - **Get Answers**: Once the PDFs are processed, you can ask questions about their content. The app will respond based on the uploaded PDFs.

## Notes

- **PDF Format**: Ensure that the uploaded files are in PDF format.
- **API Key**: The Google API key must be provided for the app to function correctly. Without it, the app will not be able to generate answers.

## Troubleshooting

- If you encounter any issues with the application, ensure that all environment variables are correctly set and that your API key is valid.
- Check the logs for any errors during the execution of the Streamlit app for further debugging.



