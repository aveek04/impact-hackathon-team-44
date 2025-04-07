import os
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import requests
import webbrowser
import time
import sys
import queue
import threading

def check_dependencies():
    """Check if all required modules are available."""
    try:
        import librosa
        import sounddevice
        import soundfile
        import requests
        import webbrowser
        return True
    except ImportError as e:
        print(f"Error: Required module not found - {str(e)}")
        print("Please install required modules using: pip install librosa sounddevice soundfile requests")
        return False

def record_audio(duration=5, sample_rate=22050):
    """Record audio from microphone."""
    print(f"\nRecording audio for {duration} seconds...")
    print("Please speak now...")
    
    # Create a queue to store the audio data
    q = queue.Queue()
    
    def callback(indata, frames, time, status):
        """Callback function to store audio data in queue."""
        if status:
            print(f"Status: {status}")
        q.put(indata.copy())
    
    # Start recording
    with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback):
        sd.sleep(int(duration * 1000))
    
    # Get the recorded data
    data = []
    while not q.empty():
        data.append(q.get())
    
    # Convert to numpy array
    audio_data = np.concatenate(data, axis=0)
    return audio_data, sample_rate

def save_audio(audio_data, sample_rate, filename="temp_recording.wav"):
    """Save recorded audio to a file."""
    sf.write(filename, audio_data, sample_rate)
    return filename

def extract_features(audio_data, sample_rate):
    """Extract audio features from the recorded data."""
    try:
        # Extract features
        # 1. Pitch (using YIN algorithm)
        pitch = librosa.yin(audio_data.flatten(), fmin=75, fmax=300)
        
        # 2. Energy (RMS)
        energy = librosa.feature.rms(y=audio_data.flatten())[0]
        
        # 3. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio_data.flatten())[0]
        
        # 4. Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data.flatten(), sr=sample_rate)[0]
        
        # Calculate mean values for each feature
        features = {
            'pitch_mean': np.mean(pitch),
            'energy_mean': np.mean(energy),
            'zcr_mean': np.mean(zcr),
            'spectral_centroid_mean': np.mean(spectral_centroid)
        }
        
        return features
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        return None

def detect_stress(audio_data, sample_rate):
    """Detect stress in the audio data based on extracted features."""
    try:
        features = extract_features(audio_data, sample_rate)
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
        if features['pitch_mean'] > 150:
            stress_indicators += 1
            print("✓ High pitch detected")
        
        # Check energy (higher energy often indicates stress)
        if features['energy_mean'] > 0.7:
            stress_indicators += 1
            print("✓ High energy detected")
        
        # Check zero crossing rate (higher ZCR often indicates stress)
        if features['zcr_mean'] > 0.1:
            stress_indicators += 1
            print("✓ High zero crossing rate detected")
        
        # Check spectral centroid (higher values often indicate stress)
        if features['spectral_centroid_mean'] > 2000:
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
    """Main function to run the microphone stress detection application."""
    print("Starting Microphone Stress Detection Application...")
    
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    while True:
        try:
            # Record audio
            audio_data, sample_rate = record_audio(duration=5)
            
            # Save the recording (optional, for debugging)
            temp_file = save_audio(audio_data, sample_rate)
            
            # Detect stress in audio
            print("\nAnalyzing audio for stress...")
            stress_detected = detect_stress(audio_data, sample_rate)
            
            if stress_detected:
                print("\n⚠️ STRESS DETECTED in your voice! ⚠️")
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
                print("\n✅ No significant stress detected in your voice.")
                print("Google Maps will not be opened as no stress was detected.")
            
            # Ask if user wants to continue
            choice = input("\nDo you want to record another sample? (y/n): ")
            if choice.lower() != 'y':
                break
                
        except KeyboardInterrupt:
            print("\nRecording stopped by user.")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            break
    
    # Clean up temporary file
    if os.path.exists("temp_recording.wav"):
        os.remove("temp_recording.wav")
    
    print("\nThank you for using the Microphone Stress Detection Application!")

if __name__ == "__main__":
    main() 