// App.js
import React, { useState } from 'react';
import { View, Text, Button, Image, StyleSheet, ActivityIndicator, ScrollView } from 'react-native';
import axios from 'axios';
import { launchImageLibrary } from 'react-native-image-picker';

const App = () => {
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [prediction, setPrediction] = useState('');
  const [maskImage, setMaskImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const pickImage = () => {
    launchImageLibrary({ mediaType: 'photo' }, (response) => {
      if (response.didCancel || response.errorCode || !response.assets) {
        console.warn('Image selection was cancelled or failed');
        return;
      }
      const uri = response.assets[0].uri;
      if (uri) {
        setImageUri(uri);
        uploadImage(uri);
      }
    });
  };

  const uploadImage = async (uri: string) => {
    setLoading(true);
    const formData = new FormData();
    formData.append('file', {
      uri: uri,
      name: 'image.png',
      type: 'image/png',
    } as any);

    try {
      const response = await axios.post('https://brecan.onrender.com/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      const { prediction, mask_image } = response.data;
      setPrediction(prediction);
      setMaskImage(`data:image/png;base64,${mask_image}`);
    } catch (error) {
      console.error('Error uploading image:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Breast Cancer Prediction</Text>
      <Button title="Upload X-ray Image" onPress={pickImage} />

      {imageUri && <Image source={{ uri: imageUri }} style={styles.image} />}

      {loading ? (
        <ActivityIndicator size="large" color="#0000ff" />
      ) : (
        prediction && (
          <View style={styles.resultContainer}>
            <Text style={styles.prediction}>Prediction: {prediction}</Text>
            {maskImage && <Image source={{ uri: maskImage }} style={styles.image} />}
          </View>
        )
      )}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  title: {
    fontSize: 24,
    marginBottom: 20,
    fontWeight: 'bold',
  },
  image: {
    width: 300,
    height: 300,
    marginVertical: 20,
    borderRadius: 10,
  },
  prediction: {
    fontSize: 18,
    marginTop: 20,
    color: '#333',
  },
  resultContainer: {
    alignItems: 'center',
    marginTop: 20,
  },
});

export default App;
