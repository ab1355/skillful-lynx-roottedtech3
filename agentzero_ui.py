import os
import json
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from hr_data import load_hr_data, calculate_avg_performance, get_dept_distribution, get_age_distribution, get_tenure_distribution
import config

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='app.log',
                    filemode='w')

# Load HR data
hr_data = load_hr_data(config.DATA_FILE)

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# In-memory storage for chat messages
chat_messages = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    logging.debug("Accessing home route")
    return render_template('index.html')

@app.route('/chat')
def chat():
    logging.debug("Accessing chat route")
    return render_template('chat.html')

def generate_response(message):
    # More sophisticated response generation
    if 'performance' in message.lower():
        avg_performance = calculate_avg_performance(hr_data)
        return f"The average performance score is {avg_performance:.2f}."
    elif 'department' in message.lower():
        dept_dist = get_dept_distribution(hr_data)
        return f"Department distribution: {dept_dist}"
    elif 'age' in message.lower():
        age_dist = get_age_distribution(hr_data)
        return f"Age distribution: {age_dist}"
    elif 'tenure' in message.lower():
        tenure_dist = get_tenure_distribution(hr_data)
        return f"Tenure distribution: {tenure_dist}"
    else:
        return f"Thank you for your message about '{message}'. How can I assist you with HR data analysis?"

@app.route('/api/messages', methods=['GET', 'POST'])
def handle_messages():
    logging.debug(f"Handling {request.method} request to /api/messages")
    if request.method == 'POST':
        logging.debug(f"POST request data: {request.form}")
        logging.debug(f"POST request files: {request.files}")
        message = request.form.get('message')
        file = request.files.get('file')
        
        logging.debug(f"Received message: {message}")
        if file:
            logging.debug(f"Received file: {file.filename}")
        
        try:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                chat_messages.append({'text': message, 'file': filename, 'sender': 'user'})
            else:
                chat_messages.append({'text': message, 'sender': 'user'})
            
            # Generate and add response
            response = generate_response(message)
            chat_messages.append({'text': response, 'sender': 'bot'})
            
            logging.debug(f"Current chat messages: {chat_messages}")
            return jsonify({'status': 'success'})
        except Exception as e:
            logging.error(f"Error processing message: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    else:
        logging.debug(f"Returning chat messages: {chat_messages}")
        return jsonify(chat_messages)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    logging.debug(f"Accessing file: {filename}")
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/visualizations')
def visualizations():
    logging.debug("Accessing visualizations route")
    dept_dist = get_dept_distribution(hr_data)
    age_dist = get_age_distribution(hr_data)
    tenure_dist = get_tenure_distribution(hr_data)
    avg_performance = calculate_avg_performance(hr_data)
    
    return render_template('visualizations.html', 
                           dept_dist=dept_dist, 
                           age_dist=age_dist, 
                           tenure_dist=tenure_dist, 
                           avg_performance=avg_performance)

if __name__ == '__main__':
    app.run(host=config.HOST, port=config.PORT, debug=True)