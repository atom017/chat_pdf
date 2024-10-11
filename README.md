# Chat with PDF App

## Demo
You can see a live demo of the app [here](https://chat-with-pdf-file.vercel.app/) which is deployed on vercel.

## Overview

The **Chat with PDF** app allows users to upload PDF documents and interact with their content through a conversational interface. By leveraging advanced natural language processing, users can ask questions about the uploaded PDFs and receive insightful answers based on the document text.

## Features

- **PDF Upload**: Easily upload PDF files for processing.
- **Text Extraction**: Extracts text from PDFs for querying.
- **Conversational Interface**: Engage with the app by asking questions about the PDF content.
- **Text Chunking**: Efficiently splits large texts into manageable chunks for processing.
- **Intelligent Responses**: Utilizes a language model to generate insightful and concise answers.

## Technologies Used

- **Flask**: Web framework for building the application.
- **PyMuPDF (fitz)**: Library for extracting text from PDF files.
- **LangChain**: Framework for managing interactions with language models.
- **ChatGroq**: Integration with Groq's language model.
- **dotenv**: For managing environment variables.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Required Python packages:
  - Flask
  - PyMuPDF
  - langchain
  - langchain-community
  - langchain_groq
  - python-dotenv

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/atom017/chat_pdf.git
   cd chat_pdf
   ```

2. **Create virtual environment (Windows)**
    ```
    python -m venv venv
    ```
3. **Activate virtual environment (Windows)**
    ```
    ./venv/Scripts/activate
    ```

4. **Install the requirements**:
    ```
    pip install -r requirements.txt
    ```

5. **Setup environment variables**:
    ```
    GROQ_API_KEY=your_api_key_here
    ```

Note: Please set up virtual environment according to your OS. 

### Running the app

1. **Start the server**:
    ```
    python app.py
    ```
2. **Open your web browser and navigate to http://127.0.0.1:5000**


### Usage

1. Upload a PDF file using the provided interface.
2. Once the file is updated, enter questions related to the content of the PDF
3. Receive answers generated by the language model based on the document text.
4. File size musth be less than 16 MB.