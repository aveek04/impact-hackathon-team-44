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
import json
import subprocess
import platform
from pynput import mouse, keyboard
from datetime import datetime

# Google Maps API Key
GOOGLE_MAPS_API_KEY = "AIzaSyCurH-0WJ2lMW34P3DJff68STAkqkf3Sow"

# Global variables for motion tracking
motion_data = {
    'mouse_movements': 0,
    'mouse_clicks': 0,
    'key_presses': 0,
    'start_time': None,
    'end_time': None
}

def check_dependencies():
    """Check if all required modules are available."""
    try:
        import librosa
        import sounddevice
        import soundfile
        import requests
        import webbrowser
        import pynput
        return True
    except ImportError as e:
        print(f"Error: Required module not found - {str(e)}")
        print("Please install required modules using: pip install librosa sounddevice soundfile requests pynput")
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

def get_location_by_geocoder():
    """Get location information using geocoder library."""
    try:
        print("Attempting to get location using geocoder...")
        
        # Try to import geocoder
        try:
            import geocoder
        except ImportError:
            print("geocoder library not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "geocoder"])
            import geocoder
        
        # Get location using geocoder
        g = geocoder.ip('me')
        
        if g.ok:
            # Extract location details
            lat = g.lat
            lng = g.lng
            city = g.city
            state = g.state
            country = g.country
            
            # Print location details
            print("\n=== LOCATION INFORMATION (Geocoder) ===")
            print("\n--- LOCATION DETAILS ---")
            print(f"CITY: {city}")
            print(f"STATE/REGION: {state}")
            print(f"COUNTRY: {country}")
            print(f"COORDINATES: {lat},{lng}")
            print("========================\n")
            
            return {
                'city': city,
                'region': state,
                'country': country,
                'loc': f"{lat},{lng}",
                'ip': g.ip,
                'org': "N/A",
                'timezone': "N/A"
            }
        else:
            print("Failed to get location using geocoder. Falling back to IP-based location.")
            return get_location_by_ip()
            
    except Exception as e:
        print(f"Error getting location from geocoder: {str(e)}")
        print("Falling back to IP-based location.")
        return get_location_by_ip()

def get_gps_location():
    """Get location information using GPS."""
    try:
        print("Attempting to get precise location using GPS...")
        
        # Check if we're on a mobile device or have access to GPS
        if platform.system() == "Windows":
            # On Windows, we'll use a simple HTML file with JavaScript to get location
            return get_location_by_google_maps()
        elif platform.system() == "Linux":
            # On Linux, try to use gpsd if available
            try:
                import gpsd
                gpsd.connect()
                packet = gpsd.get_current()
                lat, lng = packet.lat, packet.lon
                print(f"GPS coordinates: {lat}, {lng}")
                
                # Use Google Maps Geocoding API to get address details
                return get_address_from_coordinates(lat, lng)
            except ImportError:
                print("gpsd not available. Falling back to browser geolocation.")
                return get_location_by_google_maps()
        elif platform.system() == "Darwin":  # macOS
            # On macOS, we can try to use the CoreLocation framework
            try:
                # Create a simple AppleScript to get location
                script = '''
                tell application "System Events"
                    set locationData to do shell script "CoreLocationCLI -once"
                    return locationData
                end tell
                '''
                result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
                if result.returncode == 0:
                    # Parse the location data
                    location_data = result.stdout.strip().split(',')
                    if len(location_data) >= 2:
                        lat, lng = float(location_data[0]), float(location_data[1])
                        print(f"GPS coordinates: {lat}, {lng}")
                        return get_address_from_coordinates(lat, lng)
            except Exception as e:
                print(f"Error getting location from CoreLocation: {str(e)}")
            
            # Fall back to browser geolocation
            return get_location_by_google_maps()
        else:
            # For other platforms, use browser geolocation
            return get_location_by_google_maps()
            
    except Exception as e:
        print(f"Error getting GPS location: {str(e)}")
        print("Falling back to IP-based location.")
        return get_location_by_ip()

def get_address_from_coordinates(lat, lng):
    """Get address details from coordinates using Google Maps Geocoding API."""
    try:
        # Use Google Maps Geocoding API to get address details
        geocoding_url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={GOOGLE_MAPS_API_KEY}"
        response = requests.get(geocoding_url)
        data = response.json()
        
        if data["status"] != "OK":
            print(f"Geocoding API error: {data['status']}. Falling back to IP-based location.")
            return get_location_by_ip()
        
        # Extract address components
        address_components = data["results"][0]["address_components"]
        
        city = "N/A"
        region = "N/A"
        country = "N/A"
        
        for component in address_components:
            types = component["types"]
            if "locality" in types or "administrative_area_level_2" in types:
                city = component["long_name"]
            elif "administrative_area_level_1" in types:
                region = component["long_name"]
            elif "country" in types:
                country = component["long_name"]
        
        # Format location data
        location_data = {
            'city': city,
            'region': region,
            'country': country,
            'loc': f"{lat},{lng}",
            'ip': "N/A",
            'org': "N/A",
            'timezone': "N/A"
        }
        
        # Print location details
        print("\n=== LOCATION INFORMATION (GPS) ===")
        print("\n--- LOCATION DETAILS ---")
        print(f"CITY: {city}")
        print(f"STATE/REGION: {region}")
        print(f"COUNTRY: {country}")
        print(f"COORDINATES: {lat},{lng}")
        print("========================\n")
        
        return location_data
        
    except Exception as e:
        print(f"Error getting address from coordinates: {str(e)}")
        print("Falling back to IP-based location.")
        return get_location_by_ip()

def get_location_by_google_maps():
    """Get location information using Google Maps Geolocation API."""
    try:
        print("Attempting to get precise location using Google Maps API...")
        
        # Create a simple HTML file with JavaScript to get location
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Getting Location</title>
            <script>
                function getLocation() {
                    if (navigator.geolocation) {
                        navigator.geolocation.getCurrentPosition(showPosition, showError, {
                            enableHighAccuracy: true,
                            timeout: 10000,
                            maximumAge: 0
                        });
                    } else {
                        document.getElementById("result").innerHTML = "Geolocation is not supported by this browser.";
                    }
                }
                
                function showPosition(position) {
                    document.getElementById("result").innerHTML = 
                        position.coords.latitude + "," + position.coords.longitude;
                    
                    // Automatically copy coordinates to clipboard
                    const coords = position.coords.latitude + "," + position.coords.longitude;
                    navigator.clipboard.writeText(coords).then(() => {
                        document.getElementById("status").innerHTML = "Coordinates copied to clipboard!";
                    });
                }
                
                function showError(error) {
                    switch(error.code) {
                        case error.PERMISSION_DENIED:
                            document.getElementById("result").innerHTML = "User denied the request for Geolocation.";
                            break;
                        case error.POSITION_UNAVAILABLE:
                            document.getElementById("result").innerHTML = "Location information is unavailable.";
                            break;
                        case error.TIMEOUT:
                            document.getElementById("result").innerHTML = "The request to get user location timed out.";
                            break;
                        case error.UNKNOWN_ERROR:
                            document.getElementById("result").innerHTML = "An unknown error occurred.";
                            break;
                    }
                }
                
                window.onload = getLocation;
            </script>
        </head>
        <body>
            <div id="result">Getting your location...</div>
            <div id="status"></div>
        </body>
        </html>
        """
        
        # Save the HTML file
        with open("get_location.html", "w") as f:
            f.write(html_content)
        
        # Open the HTML file in the default browser
        webbrowser.open("file://" + os.path.abspath("get_location.html"))
        
        # Wait for user to input coordinates
        print("\nPlease check the browser window that opened.")
        print("The coordinates have been automatically copied to your clipboard.")
        print("Paste the coordinates below (or press Enter to use clipboard content):")
        coordinates = input("Coordinates (e.g., 12.9719,77.5937): ")
        
        # If user just pressed Enter, try to get from clipboard
        if not coordinates:
            try:
                import pyperclip
                coordinates = pyperclip.paste()
                print(f"Using coordinates from clipboard: {coordinates}")
            except ImportError:
                print("pyperclip not installed. Please paste the coordinates manually.")
                coordinates = input("Coordinates: ")
        
        # Clean up the HTML file
        if os.path.exists("get_location.html"):
            os.remove("get_location.html")
        
        if not coordinates or "," not in coordinates:
            print("Invalid coordinates. Falling back to IP-based location.")
            return get_location_by_ip()
        
        # Parse coordinates
        lat, lng = coordinates.split(",")
        lat = float(lat.strip())
        lng = float(lng.strip())
        
        # Get address from coordinates
        return get_address_from_coordinates(lat, lng)
        
    except Exception as e:
        print(f"Error getting location from Google Maps: {str(e)}")
        print("Falling back to IP-based location.")
        return get_location_by_ip()

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
        maps_url = f"https://www.google.com/maps/search/?api=1&query={lat},{lng}&key={GOOGLE_MAPS_API_KEY}&zoom=15"
        
        # Open in default browser
        print(f"\nOpening Google Maps for: {location_data['city']}, {location_data['region']}, {location_data['country']}")
        print(f"Coordinates: {lat}, {lng}")
        print(f"URL: {maps_url}\n")
        
        webbrowser.open(maps_url)
        return True
    except Exception as e:
        print(f"Error opening Google Maps: {str(e)}")
        return False

def start_motion_tracking():
    """Start tracking mouse and keyboard activity."""
    global motion_data
    motion_data = {
        'mouse_movements': 0,
        'mouse_clicks': 0,
        'key_presses': 0,
        'start_time': datetime.now(),
        'end_time': None
    }
    
    # Create mouse listener
    mouse_listener = mouse.Listener(
        on_move=lambda x, y: track_mouse_movement(),
        on_click=lambda x, y, button, pressed: track_mouse_click(pressed)
    )
    
    # Create keyboard listener
    keyboard_listener = keyboard.Listener(
        on_press=lambda key: track_key_press()
    )
    
    # Start listeners
    mouse_listener.start()
    keyboard_listener.start()
    
    return mouse_listener, keyboard_listener

def stop_motion_tracking(mouse_listener, keyboard_listener):
    """Stop tracking mouse and keyboard activity."""
    global motion_data
    motion_data['end_time'] = datetime.now()
    
    # Stop listeners
    mouse_listener.stop()
    keyboard_listener.stop()
    
    # Calculate duration in seconds
    duration = (motion_data['end_time'] - motion_data['start_time']).total_seconds()
    
    # Calculate rates per second
    motion_data['mouse_movement_rate'] = motion_data['mouse_movements'] / duration if duration > 0 else 0
    motion_data['mouse_click_rate'] = motion_data['mouse_clicks'] / duration if duration > 0 else 0
    motion_data['key_press_rate'] = motion_data['key_presses'] / duration if duration > 0 else 0
    
    return motion_data

def track_mouse_movement():
    """Track mouse movement events."""
    global motion_data
    motion_data['mouse_movements'] += 1

def track_mouse_click(pressed):
    """Track mouse click events."""
    global motion_data
    if pressed:  # Only count press events, not releases
        motion_data['mouse_clicks'] += 1

def track_key_press():
    """Track keyboard press events."""
    global motion_data
    motion_data['key_presses'] += 1

def analyze_motion_data(motion_data):
    """Analyze motion data to detect stress indicators."""
    # Define thresholds for stress indicators
    high_movement_threshold = 10  # movements per second
    high_click_threshold = 2      # clicks per second
    high_keypress_threshold = 5   # keypresses per second
    
    stress_indicators = 0
    
    # Check mouse movement rate
    if motion_data['mouse_movement_rate'] > high_movement_threshold:
        stress_indicators += 1
        print("✓ High mouse movement rate detected")
    
    # Check mouse click rate
    if motion_data['mouse_click_rate'] > high_click_threshold:
        stress_indicators += 1
        print("✓ High mouse click rate detected")
    
    # Check key press rate
    if motion_data['key_press_rate'] > high_keypress_threshold:
        stress_indicators += 1
        print("✓ High keyboard activity detected")
    
    # Print motion data for debugging
    print("\nMotion Data:")
    print(f"Mouse Movements: {motion_data['mouse_movements']} ({motion_data['mouse_movement_rate']:.2f}/sec)")
    print(f"Mouse Clicks: {motion_data['mouse_clicks']} ({motion_data['mouse_click_rate']:.2f}/sec)")
    print(f"Key Presses: {motion_data['key_presses']} ({motion_data['key_press_rate']:.2f}/sec)")
    
    # Detect stress if at least 1 indicator is present
    is_stressed = stress_indicators >= 1
    print(f"\nTotal motion stress indicators: {stress_indicators}/3")
    return is_stressed

def main():
    """Main function to run the microphone stress detection application."""
    print("Starting Microphone Stress Detection Application...")
    
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    while True:
        try:
            # Start motion tracking
            print("\nStarting motion tracking...")
            mouse_listener, keyboard_listener = start_motion_tracking()
            
            # Record audio
            audio_data, sample_rate = record_audio(duration=5)
            
            # Stop motion tracking
            print("\nStopping motion tracking...")
            motion_data = stop_motion_tracking(mouse_listener, keyboard_listener)
            
            # Save the recording (optional, for debugging)
            temp_file = save_audio(audio_data, sample_rate)
            
            # Detect stress in audio
            print("\nAnalyzing audio for stress...")
            audio_stress_detected = detect_stress(audio_data, sample_rate)
            
            # Analyze motion data
            print("\nAnalyzing motion data for stress...")
            motion_stress_detected = analyze_motion_data(motion_data)
            
            # Combined stress detection
            stress_detected = audio_stress_detected or motion_stress_detected
            
            if stress_detected:
                print("\n⚠️ STRESS DETECTED! ⚠️")
                if audio_stress_detected:
                    print("- Stress detected in your voice")
                if motion_stress_detected:
                    print("- Stress detected in your physical activity")
                
                print("Opening Google Maps to help you find nearby relaxation spots...")
                
                # Get location data using geocoder
                print("\nFetching your location...")
                location_data = get_location_by_geocoder()
                
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
                print("\n✅ No significant stress detected.")
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