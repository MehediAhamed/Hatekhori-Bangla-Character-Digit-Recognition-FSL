from flask import Flask, request, jsonify, render_template, send_from_directory
import subprocess
import torch
from torchvision import transforms
from PIL import Image
import io
import os
import model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains on all routes

subprocess.Popen(["python", "model.py"])

# Load your trained model (ensure you have the model saved as a .pth file)
# model = torch.load('path_to_your_model.pth', map_location=torch.device('cpu'))  # Load the model on CPU if you are not using GPU
# model.eval()  # Set the model to evaluation mode

# Define the mapping of model output indices to Bengali characters
label_to_bengali = {
    1: "অ", 2: "আ", 3: "ই", 4: "ঈ", 5: "উ", 6: "ঊ", 7: "ঋ", 8: "এ", 9: "ঐ", 10: "ও", 11: "ঔ",
    12: "ক", 13: "খ", 14: "গ", 15: "ঘ", 16: "ঙ", 17: "চ", 18: "ছ", 19: "জ", 20: "ঝ", 21: "ঞ",
    22: "ট", 23: "ঠ", 24: "ড", 25: "ঢ", 26: "ণ", 27: "ত", 28: "থ", 29: "দ", 30: "ধ", 31: "ন",
    32: "প", 33: "ফ", 34: "ব", 35: "ভ", 36: "ম", 37: "য", 38: "র", 39: "ল", 40: "শ", 41: "ষ",
    42: "স", 43: "হ", 44: "ড়", 45: "ঢ়", 46: "য়", 47: "ৎ", 48: "ং", 49: "ঃ", 50: "ঁ",
    51: "০", 52: "১", 53: "২", 54: "৩", 55: "৪", 56: "৫", 57: "৬", 58: "৭", 59: "৮", 60: "৯"
}

# Define the image preprocessing
image_transforms = transforms.Compose([
    transforms.Resize((84, 84)),  # Resize the image to what the model expects
    transforms.ToTensor(),        # Convert the image to a tensor
])

# Ensure the content directory exists
if not os.path.exists('content'):
    os.makedirs('content')

@app.route('/')
def index():
    return render_template('index.html')
 
@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory('static/audio', filename)


@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/instruction')
def instruction():
    return render_template('instructions.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Save the file to the content folder
        file_path = os.path.join('content', 'drawing.jpg')
        file.save(file_path)
        
        # Read the saved image and preprocess it for the model
        img = Image.open(file_path).convert('RGB')
        img_tensor = image_transforms(img).unsqueeze(0)  # Add batch dimension

        # Use the model to predict
        
        query_image_path = 'content/drawing.jpg'
        predicted_label = model.predict_image_with_support_set(query_image_path)
        print("Predicted label: ")
        label = model.print_bengali_character(predicted_label)

        
        # with torch.no_grad():
        #     outputs = model(img_tensor)
        #     probabilities = torch.softmax(outputs, dim=1)
        #     predicted_label_idx = probabilities.argmax(1).item()
        #     predicted_label = predicted_label_idx + 1  # Adjust label to match original range
        #     predicted_char = label_to_bengali[predicted_label]

        # return jsonify({'extracted_text': predicted_char})
        
        return jsonify({'extracted_text': label, 'label': predicted_label})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
