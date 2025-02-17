import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class BreastCancerDataset(Dataset):
    def __init__(self, image_paths, mask_paths, labels=None, image_transform=None, mask_transform=None, return_labels=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.labels = labels
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.return_labels = return_labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        if self.return_labels and self.labels is not None:
            label = torch.tensor(self.labels[idx])
            return image, mask, label
        return image, mask


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Decoder
        self.up3 = self.up_conv(512, 256)
        self.dec3 = self.conv_block(512, 256)
        self.up2 = self.up_conv(256, 128)
        self.dec2 = self.conv_block(256, 128)
        self.up1 = self.up_conv(128, 64)
        self.dec1 = self.conv_block(128, 64)
        
        self.final = nn.Conv2d(64, 1, kernel_size=1)
        
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def up_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(nn.MaxPool2d(2)(e1))
        e3 = self.enc3(nn.MaxPool2d(2)(e2))
        e4 = self.enc4(nn.MaxPool2d(2)(e3))
        
        # Decoder
        d3 = self.up3(e4)
        d3 = self.dec3(torch.cat([d3, e3], 1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], 1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], 1))
        
        return torch.sigmoid(self.final(d1))

class Classifier(nn.Module):
    def __init__(self, num_classes=3):
        super(Classifier, self).__init__()
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=True)
        
        # Modify first conv layer to accept 4 channels (RGB + mask)
        original_layer = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Initialize the new layer with pretrained weights for RGB channels
        with torch.no_grad():
            self.backbone.conv1.weight[:, :3] = original_layer.weight
            self.backbone.conv1.weight[:, 3] = original_layer.weight.mean(dim=1)
        
        # Modify final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, mask):
        # Concatenate image and mask
        x = torch.cat([x, mask], dim=1)
        return self.backbone(x)

class BreastCancerSystem:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mask_generator = UNet().to(self.device)
        self.classifier = Classifier().to(self.device)
        
        # Separate transforms for images and masks
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def prepare_data(self, data_dir):
        image_paths = []
        mask_paths = []
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
                mask_path = os.path.join(category_dir, img_name.replace('.png', '_mask.png'))
                
                if os.path.exists(mask_path):
                    image_paths.append(img_path)
                    mask_paths.append(mask_path)
                    labels.append(label_map[category])

        return train_test_split(image_paths, mask_paths, labels, test_size=0.2, random_state=42)

    def train_mask_generator(self, train_loader, val_loader, num_epochs=50):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.mask_generator.parameters(), lr=0.001)
        best_loss = float('inf')

        for epoch in range(num_epochs):
            self.mask_generator.train()
            train_loss = 0
            
            for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.mask_generator(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()

            # Validation
            val_loss = self.evaluate_mask_generator(val_loader)
            print(f'Epoch {epoch+1}/{num_epochs}')
            print(f'Training Loss: {train_loss/len(train_loader):.4f}')
            print(f'Validation Loss: {val_loss:.4f}')

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.mask_generator.state_dict(), 'best_mask_generator.pth')

    def train_classifier(self, train_loader, val_loader, num_epochs=50):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.classifier.parameters(), lr=0.001)
        best_acc = 0

        for epoch in range(num_epochs):
            self.classifier.train()
            train_loss = 0
            correct = 0
            total = 0

            for images, masks, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
                images = images.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.classifier(images, masks)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            # Validation
            val_acc = self.evaluate_classifier(val_loader)
            print(f'Epoch {epoch+1}/{num_epochs}')
            print(f'Training Accuracy: {100.*correct/total:.2f}%')
            print(f'Validation Accuracy: {val_acc:.2f}%')

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.classifier.state_dict(), 'best_classifier.pth')

    def evaluate_mask_generator(self, loader):
        self.mask_generator.eval()
        total_loss = 0
        criterion = nn.BCELoss()

        with torch.no_grad():
            for images, masks in loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                outputs = self.mask_generator(images)
                loss = criterion(outputs, masks)
                total_loss += loss.item()

        return total_loss / len(loader)

    def evaluate_classifier(self, loader):
        self.classifier.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, masks, labels in loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)
                outputs = self.classifier(images, masks)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return 100. * correct / total

    def predict(self, image_path):
        self.mask_generator.eval()
        self.classifier.eval()
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        transformed_image = self.image_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Generate mask
            mask = self.mask_generator(transformed_image)
            
            # Make prediction
            outputs = self.classifier(transformed_image, mask)
            _, predicted = outputs.max(1)
            
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
                verticalalignment='center')
        plt.axis('off')
        
        plt.show()
        return prediction, mask.squeeze().cpu().numpy()

def main():
    # Initialize the system
    system = BreastCancerSystem()
    
    # Prepare data
    train_images, val_images, train_masks, val_masks, train_labels, val_labels = system.prepare_data('data')
    
    # Create datasets with separate transforms
    train_dataset = BreastCancerDataset(
        train_images, 
        train_masks, 
        train_labels, 
        image_transform=system.image_transform,
        mask_transform=system.mask_transform,
        return_labels=False  # Do not return labels for mask generator
    )
    val_dataset = BreastCancerDataset(
        val_images, 
        val_masks, 
        val_labels, 
        image_transform=system.image_transform,
        mask_transform=system.mask_transform,
        return_labels=False  # Do not return labels for mask generator
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Train models
    print("Training Mask Generator...")
    system.train_mask_generator(train_loader, val_loader, num_epochs=50)
    
    # Create datasets with labels for classifier training
    train_dataset = BreastCancerDataset(
        train_images, 
        train_masks, 
        train_labels, 
        image_transform=system.image_transform,
        mask_transform=system.mask_transform,
        return_labels=True  # Return labels for classifier
    )
    val_dataset = BreastCancerDataset(
        val_images, 
        val_masks, 
        val_labels, 
        image_transform=system.image_transform,
        mask_transform=system.mask_transform,
        return_labels=True  # Return labels for classifier
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    print("Training Classifier...")
    system.train_classifier(train_loader, val_loader, num_epochs=50)
    
    # Load best models
    system.mask_generator.load_state_dict(torch.load('best_mask_generator.pth'))
    system.classifier.load_state_dict(torch.load('best_classifier.pth'))
    
    # Example prediction
    print("Testing on a random validation image...")
    random_idx = random.randint(0, len(val_images)-1)
    prediction, mask = system.predict(val_images[random_idx])
    print(f"Predicted class: {prediction}")
    print(f"True class: {['Normal', 'Benign', 'Malignant'][val_labels[random_idx]]}")

if __name__ == "__main__":
    main()