import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# Import the UNet and Classifier classes
from models import UNet, Classifier  # Assuming you moved the classes to models.py

# Load the trained models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mask_generator = UNet().to(device)
mask_generator.load_state_dict(torch.load('best_mask_generator.pth', map_location=device))
mask_generator.eval()

classifier = Classifier().to(device)
classifier.load_state_dict(torch.load('best_classifier.pth', map_location=device))
classifier.eval()

# Define image transformations
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Function to make predictions
def predict(image_path):
    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Generate mask
        mask = mask_generator(image_tensor)
        
        # Make prediction
        output = classifier(image_tensor, mask)
        _, predicted = output.max(1)
        classes = ['Normal', 'Benign', 'Malignant']
        prediction = classes[predicted.item()]

    # Visualize results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask.squeeze().cpu().numpy(), cmap='gray')
    plt.title('Generated Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.text(0.5, 0.5, f'Prediction: {prediction}', 
             horizontalalignment='center', 
             verticalalignment='center', 
             fontsize=12)
    plt.axis('off')

    plt.show()

    return prediction

# Function to evaluate accuracy on a dataset
def evaluate_accuracy(data_dir):
    image_paths = []
    labels = []
    label_map = {'normal': 0, 'benign': 1, 'malignant': 2}

    for category in ['normal', 'benign', 'malignant']:
        category_dir = os.path.join(data_dir, category)
        if not os.path.exists(category_dir):
            continue

        for img_name in os.listdir(category_dir):
            if img_name.endswith('_mask.png'):
                continue

            img_path = os.path.join(category_dir, img_name)
            image_paths.append(img_path)
            labels.append(label_map[category])

    correct = 0
    total = len(image_paths)

    for i, image_path in enumerate(image_paths):
        true_label = labels[i]
        predicted_label = predict(image_path)

        if predicted_label.lower() == list(label_map.keys())[list(label_map.values()).index(true_label)]:
            correct += 1

        print(f'Image: {image_path}')
        print(f'True Label: {list(label_map.keys())[list(label_map.values()).index(true_label)]}')
        print(f'Predicted Label: {predicted_label}')
        print('---')

    accuracy = 100. * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

# Main function
if __name__ == '__main__':
    # Example: Predict on a single image
    image_path = 'data/malignant/malignant (44).png'  # Replace with the path to your image
    print(f'Prediction: {predict(image_path)}')

    # Example: Evaluate accuracy on a dataset
    # data_dir = 'data'  # Replace with the path to your dataset
    # evaluate_accuracy(data_dir)