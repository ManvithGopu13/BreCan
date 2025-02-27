from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
import io
import base64
from models import UNet, Classifier  # Import the classes
import os  # Add this import at the top

app = Flask(__name__)

# Load your trained models here
mask_generator = UNet().to('cpu')
mask_generator.load_state_dict(torch.load('best_mask_generator.pth', map_location='cpu'))
mask_generator.eval()

classifier = Classifier().to('cpu')
classifier.load_state_dict(torch.load('best_classifier.pth', map_location='cpu'))
classifier.eval()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert('RGB')

    # Preprocess and predict
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to('cpu')

    with torch.no_grad():
        # Generate mask
        mask = mask_generator(image_tensor)
        
        # Make prediction
        output = classifier(image_tensor, mask)
        _, predicted = output.max(1)
        classes = ['Normal', 'Benign', 'Malignant']
        prediction = classes[predicted.item()]

        # Convert mask tensor to PIL image
        mask_image = transforms.ToPILImage()(mask.squeeze().cpu())

        # Save mask image to a bytes buffer
        mask_buffer = io.BytesIO()
        mask_image.save(mask_buffer, format="PNG")
        mask_buffer.seek(0)

        # Encode mask image as base64
        mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode('utf-8')

    return jsonify({
        'prediction': prediction,
        'mask_image': mask_base64  # Send mask image as base64
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Get PORT from environment or default to 5000
    app.run(host='0.0.0.0', port=port) # Make the server publicly available