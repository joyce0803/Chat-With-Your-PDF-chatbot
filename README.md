# Chat-with-your-PDF-chatbot #

Chat-with-Your-PDF-Chatbot is an open-source project that leverages an advanced language model to answer questions related to the content of PDF documents. This project is particularly useful for scenarios where information retrieval from PDFs is required through natural language queries.

## Features ##
- PDF Document Processing:
  The chatbot can analyze and extract information from uploaded PDF documents.
- Natural Language Understanding: Utilizes a powerful language model to comprehend and respond to questions posed in natural language.
- GPU Acceleration: Takes advantage of GPU acceleration for efficient and faster processing. (Note: GPU is recommended for optimal performance. It will run on CPU too but will take some time to display responses.).
- Google Colab Integration: A Google Colab notebook is provided for users who want to run the chatbot in a cloud environment.

## Usage ##
- Local Setup:
Ensure you have the necessary dependencies installed. (Listed in the requirements.txt file).
Run the Streamlit frontend locally using the command: streamlit run app.py.
- Google Colab: Open the provided Google Colab notebook.
Follow the instructions to upload your PDF and interact with the chatbot.
- Asking Questions:
Upload a PDF document using the provided interface.
Type your questions in natural language.
Receive answers based on the content of the PDF.

## Requirements ##
- Python 3.6 or later
- CPU or GPU
- Streamlit
- PyTorch and Hugging Face Transformers library
