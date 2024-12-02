Advanced Q&A Chatbot Project :

    This project implements an interactive Question and Answer chatbot using Python, Flask, and a simple frontend. It leverages NLP and machine learning models to generate responses based on user queries.

Project Structure:
    File Paths and Descriptions
        Backend Files

        1.advanced_qna.py
            Contains the core logic for the chatbot, including ML-based answer generation.
            Path: ./advanced_qna.py

        2.data_set.json
            A JSON file containing the dataset with pre-defined question-answer pairs.
            Path: ./data_set.json

        3.app.py
            Flask server connecting the frontend to the backend.
            Path: ./app.py

        4.Frontend Files

            templates/index.html
            The main HTML file for the chatbot interface.
            Path: ./templates/index.html

        5.static/style.css
            The CSS file for styling the chatbot interface.
            Path: ./static/style.css

        6.Embedded JavaScript
            The JavaScript code for handling user input and communicating with the Flask backend is embedded within index.html.

Step-by-Step Setup:


1. Prerequisites
Ensure the following software is installed:

Python 3.8+
Pip (Python's package manager)


2. Folder Structure
    Set up the folder structure as follows:

    Advanced_QnA_Chatbot/
    │
    ├── advanced_qna.py               # Python backend with chatbot logic
    ├── app.py               # Flask server for connecting frontend and backend
    ├── data_set.json        # Dataset of questions and answers
    │
    ├── templates/           # Frontend templates folder
    │   └── index.html       # HTML file for chatbot interface
    │
    ├── static/              # Static files folder
        └── style.css        # CSS file for styling


3. Install Required Libraries
    Run the following commands to install dependencies:

        >pip install flask numpy torch scikit-learn transformers nltk

    Download NLTK Data
        Execute the following Python snippet to download essential NLTK data:
        
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')

4. Run the Application
    Run the Flask app:

        >python app.py


    The app will start at http://127.0.0.1:5000. Open the link in your browser.

5. Interact with the Chatbot
    Enter your question in the input box.
    Press "Send."
    The chatbot will process your query and respond.

6. Customization
    Dataset: Add more question-answer pairs to data_set.json.
    Styling: Modify style.css to customize the chatbot interface.
    Models: Use a different transformer model by updating the model_name in qna.py.
