from flask import Flask, request, jsonify, render_template
import json
from advanced_qna import AdvancedQNAChatbot  # Assuming your chatbot class is in a file named advanced_qna.py

# Initialize Flask app
app = Flask(__name__)

# Initialize chatbot with your JSON file
chatbot = AdvancedQNAChatbot('data_set.json')

@app.route('/')
def index():
    return render_template('index.html')  # Assuming the HTML file is named 'index.html'

@app.route('/get', methods=['POST'])
def get_bot_response():
    user_msg = request.form.get('msg')  # Get the user's message from the POST request
    if user_msg:
        bot_response = chatbot.generate_answer(user_msg)  # Generate response using the chatbot
        return jsonify({"response": bot_response})
    return jsonify({"response": "I didn't understand that."})


if __name__ == "__main__":
    app.run(debug=True)
