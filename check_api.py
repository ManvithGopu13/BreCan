import requests
import base64

url = 'http://localhost:5000/predict'
files = {'file': open('data/normal/normal (6).png', 'rb')}
response = requests.post(url, files=files)

if response.status_code == 200:
    result = response.json()
    print('Prediction:', result['prediction'])
    
    # Decode and save the mask image
    mask_base64 = result['mask_image']
    mask_bytes = base64.b64decode(mask_base64)
    with open('mask_output.png', 'wb') as f:
        f.write(mask_bytes)
    print('Mask image saved as mask_output.png')
else:
    print('Error:', response.json())