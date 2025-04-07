import os
import pandas as pd
import librosa
import numpy as np
import requests
import webbrowser
import time
import sys

def check_dependencies():
    """Check if all required modules are available."""
    try:
        import librosa
        import pandas
        import requests
        import webbrowser
        return True
    except ImportError as e:
        print(f"Error: Required module not found - {str(e)}")
        print("Please install required modules using: pip install librosa pandas requests")
        return False

def extract_features(filepath):
    """Extract audio features from the given file."""
    try:
        # Load the audio file
        y, sr = librosa.load(filepath)
        
        # Extract features
        # 1. Pitch (using YIN algorithm)
        pitch = librosa.yin(y, fmin=75, fmax=300)
        
        # 2. Energy (RMS)
        energy = librosa.feature.rms(y=y)[0]
        
        # 3. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        # 4. Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        # Calculate mean values for each feature
        features = {
            'pitch_mean': np.mean(pitch),
            'energy_mean': np.mean(energy),
            'zcr_mean': np.mean(zcr),
            'spectral_centroid_mean': np.mean(spectral_centroid)
        }
        
        return features
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return None

def detect_stress(audio_file):
    """Detect stress in the audio file based on extracted features."""
    try:
        features = extract_features(audio_file)
        if features is None:
            return False
        
        # Print feature values for debugging
        print("\nExtracted Audio Features:")
        print(f"Pitch Mean: {features['pitch_mean']:.2f}")
        print(f"Energy Mean: {features['energy_mean']:.2f}")
        print(f"Zero Crossing Rate: {features['zcr_mean']:.2f}")
        print(f"Spectral Centroid: {features['spectral_centroid_mean']:.2f}")
        
        # Simple stress detection based on feature thresholds
        stress_indicators = 0
        
        # Check pitch (higher pitch often indicates stress)
        if features['pitch_mean'] > 150:  # Increased threshold
            stress_indicators += 1
            print("✓ High pitch detected")
        
        # Check energy (higher energy often indicates stress)
        if features['energy_mean'] > 0.7:  # Increased threshold
            stress_indicators += 1
            print("✓ High energy detected")
        
        # Check zero crossing rate (higher ZCR often indicates stress)
        if features['zcr_mean'] > 0.1:  # Increased threshold
            stress_indicators += 1
            print("✓ High zero crossing rate detected")
        
        # Check spectral centroid (higher values often indicate stress)
        if features['spectral_centroid_mean'] > 2000:  # Increased threshold
            stress_indicators += 1
            print("✓ High spectral centroid detected")
        
        # Detect stress if at least 2 indicators are present
        is_stressed = stress_indicators >= 2
        print(f"\nTotal stress indicators: {stress_indicators}/4")
        return is_stressed
        
    except Exception as e:
        print(f"Error in stress detection: {str(e)}")
        return False

def get_location_by_ip():
    """Get location information based on IP address."""
    try:
        response = requests.get('https://ipinfo.io/json')
        data = response.json()
        
        # Extract location details
        ip = data.get('ip', 'N/A')
        city = data.get('city', 'N/A')
        region = data.get('region', 'N/A')
        country = data.get('country', 'N/A')
        loc = data.get('loc', 'N/A')  # Latitude, Longitude
        org = data.get('org', 'N/A')
        timezone = data.get('timezone', 'N/A')
        
        # Print location details
        print("\n=== LOCATION INFORMATION ===")
        print(f"IP Address: {ip}")
        print(f"Organization: {org}")
        print("\n--- LOCATION DETAILS ---")
        print(f"CITY: {city}")
        print(f"STATE/REGION: {region}")
        print(f"COUNTRY: {country}")
        print(f"COORDINATES: {loc}")
        print(f"TIMEZONE: {timezone}")
        print("========================\n")
        
        return {
            'city': city,
            'region': region,
            'country': country,
            'loc': loc,
            'ip': ip,
            'org': org,
            'timezone': timezone
        }
    except requests.exceptions.RequestException as e:
        print(f"Network Error: Could not connect to location service - {str(e)}")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def open_google_maps(location_data):
    """Open Google Maps with the specified location."""
    if not location_data:
        print("Error: No location data available. Cannot open Google Maps.")
        return False
    
    if not location_data['loc']:
        print("Error: No coordinates available. Cannot open Google Maps.")
        return False
    
    try:
        # Extract latitude and longitude
        lat, lng = location_data['loc'].split(',')
        
        # Create Google Maps URL with zoom level and marker
        maps_url = f"https://www.google.com/maps/search/?api=1&query={lat},{lng}&zoom=12"
        
        # Open in default browser
        print(f"\nOpening Google Maps for: {location_data['city']}, {location_data['region']}, {location_data['country']}")
        print(f"Coordinates: {lat}, {lng}")
        print(f"URL: {maps_url}\n")
        
        webbrowser.open(maps_url)
        return True
    except Exception as e:
        print(f"Error opening Google Maps: {str(e)}")
        return False

def main():
    """Main function to run the stress detection and location application."""
    print("Starting Stress Detection and Location Application...")
    
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    # Use samplewav.wav
    audio_file = "samplewav.wav"
    
    print(f"\n{'='*50}")
    print(f"Testing audio file: {audio_file}")
    print('='*50)
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' does not exist!")
        sys.exit(1)
    
    # Detect stress in audio
    print("\nAnalyzing audio for stress...")
    stress_detected = detect_stress(audio_file)
    
    if stress_detected:
        print("\n⚠️ STRESS DETECTED in the audio! ⚠️")
        print("Opening Google Maps to help you find nearby relaxation spots...")
        
        # Get location data
        print("\nFetching your location...")
        location_data = get_location_by_ip()
        
        # Only proceed with Google Maps if location data was successfully retrieved
        if location_data:
            print("Location data retrieved successfully!")
            if open_google_maps(location_data):
                print("Google Maps opened successfully!")
            else:
                print("Failed to open Google Maps.")
        else:
            print("Failed to retrieve location data. Cannot open Google Maps.")
    else:
        print("\n✅ No significant stress detected in the audio.")
        print("Google Maps will not be opened as no stress was detected.")

if __name__ == "__main__":
    main() 