#Healthcare Chatbot System

Overview

The Healthcare Chatbot System is an intelligent conversational AI designed to assist users in addressing their healthcare-related queries. This chatbot leverages natural language processing and machine learning techniques to provide accurate responses and guidance, offering a scalable and efficient solution for healthcare communication.

Features

Intuitive Interface: Easy-to-use interface for seamless interaction.

Smart Responses: Provides answers to common healthcare queries using trained machine learning models.

Customizable Intents: Adaptable to different healthcare domains through custom intent configurations.

Database Integration: Stores user interactions and data in a secure SQLite database.

Lightweight Deployment: Developed with Python and Flask for simple deployment.

Project Structure

Healthcare-Chatbot-System/
├── health/                 # Contains training and configuration files
├── images/                 # Stores logo and visual assets
├── static/                 # Static files (CSS, JavaScript)
├── templates/              # HTML templates for the web interface
├── Health.h5               # Trained model file
├── Health.py               # Model training script
├── app.py                  # Flask application entry point
├── db.sqlite3              # SQLite database for storing interactions
├── file.txt                # Additional project notes or data
├── healthlogo.jpg          # Logo image
├── intents.json            # Intents and responses for chatbot training
├── intents_short.json      # Simplified intents configuration

Installation

Clone the Repository:

git clone https://github.com/brendon-tk/Healthcare-Chatbot-System.git
cd Healthcare-Chatbot-System

Install Dependencies:

pip install -r requirements.txt

Run the Application:

python app.py

How It Works


![WhatsApp Image 2024-12-17 at 14 43 05_53690bbf](https://github.com/user-attachments/assets/7e5e2543-c94f-445e-9943-6a0d08cf13ad)


![WhatsApp Image 2024-12-17 at 14 43 10_b0e721aa](https://github.com/user-attachments/assets/b118b1cc-8805-43ca-b655-b3b347288740)

User Interaction:

Users interact with the chatbot via a web interface.

Intent Matching:

The chatbot processes user input and matches it with predefined intents from intents.json.

Response Generation:

The trained machine learning model (Health.h5) generates responses based on the matched intent.

Data Storage:

User interactions are logged into the SQLite database (db.sqlite3) for analysis and improvement.

Customization

Updating Intents

Modify intents.json to add or change intents and responses.

Train the model again using Health.py.

Training the Model

Run the following command to retrain the model:

python Health.py

Technologies Used

Programming Language: Python

Web Framework: Flask

Machine Learning: TensorFlow/Keras

Database: SQLite

Frontend: HTML, CSS, JavaScript

Contributions

Contributions are welcome! Feel free to fork the repository, make improvements, and create pull requests. Please ensure your code adheres to the project's style and standards.

License

This project is licensed under the MIT License. Feel free to use and adapt it for your needs.

Author

Brendon TK

For inquiries or support, please contact: brendonmatsikinya@gmail.com

Acknowledgments

Inspired by advancements in conversational AI for healthcare.

Special thanks to open-source contributors and the machine learning community.

