import os
import sys
import time
import threading
import argparse
from datetime import datetime
import queue
import numpy as np
import sounddevice as sd

# Import modules from the original stress detector
from mic_stress_detector_hackathon import (
    check_dependencies, record_audio, save_audio, detect_stress,
    get_location_by_geocoder, open_google_maps
)

# Import the camera motion detector
from camera_motion_detector import CameraMotionDetector

class IntegratedStressDetector:
    """
    A class that integrates microphone-based voice stress detection with
    camera-based motion stress detection for a more comprehensive stress analysis.
    Focused on detecting panic and high-stress situations only.
    """
    
    def __init__(self, camera_id=0, audio_duration=5, motion_duration=5):
        """
        Initialize the integrated stress detector.
        
        Args:
            camera_id (int): The ID of the camera to use (0 for default/front camera)
            audio_duration (int): Duration of audio recording in seconds
            motion_duration (int): Duration of motion detection in seconds
        """
        self.camera_id = camera_id
        self.audio_duration = audio_duration
        self.motion_duration = motion_duration
        
        # Initialize camera motion detector with higher threshold for panic detection
        self.motion_detector = CameraMotionDetector(
            camera_id=self.camera_id,
            recording_duration=self.motion_duration,
            motion_threshold=30,  # Increased threshold for more selective detection
            stress_threshold=0.5  # Only trigger for significant stress/panic
        )
        
        # Results storage
        self.results = {
            'audio_stress_detected': False,
            'motion_stress_detected': False,
            'audio_stress_level': 0.0,
            'motion_stress_level': 0.0,
            'combined_stress_level': 0.0,
            'panic_detected': False,
            'location_data': None
        }
        
        # Thresholds for panic detection
        self.panic_threshold = 0.7  # Combined stress level threshold for panic
        self.high_stress_threshold = 0.5  # Threshold for high stress
    
    def analyze_combined_stress(self, audio_stress_detected, audio_stress_level, motion_stress_level):
        """
        Analyze the combined stress levels to determine if panic/high-stress is present.
        
        Args:
            audio_stress_detected (bool): Whether stress was detected in audio
            audio_stress_level (float): The stress level detected from audio (0.0 to 1.0)
            motion_stress_level (float): The stress level detected from motion (0.0 to 1.0)
            
        Returns:
            tuple: (is_panic, is_high_stress, combined_level, description)
        """
        # Calculate combined stress level with weighted average
        # Give slightly more weight to motion for panic detection
        combined_level = (audio_stress_level * 0.4) + (motion_stress_level * 0.6)
        
        # Determine stress state
        is_panic = combined_level >= self.panic_threshold
        is_high_stress = combined_level >= self.high_stress_threshold
        
        # Generate description
        description = "Calm" if combined_level < 0.3 else \
                     "Mild Stress" if combined_level < 0.5 else \
                     "Moderate Stress" if combined_level < 0.7 else \
                     "High Stress" if combined_level < 0.9 else "Panic"
        
        return is_panic, is_high_stress, combined_level, description
    
    def run_detection(self, record_video=False):
        """Run the integrated stress detection process."""
        print("\n=== STARTING PANIC/HIGH-STRESS DETECTION ===")
        print("This will analyze your voice and physical movements for signs of panic or high stress.")
        
        # Start motion detection in a separate thread
        print("\nStarting motion detection...")
        self.motion_detector.start_detection()
        
        # Record video if requested
        if record_video:
            video_thread = threading.Thread(target=self.motion_detector.record_video)
            video_thread.daemon = True
            video_thread.start()
        
        # Record and analyze audio
        print("\nRecording audio...")
        audio_data, sample_rate = self.record_audio()
        temp_file = save_audio(audio_data, sample_rate)
        
        # Wait for motion detection to complete
        time.sleep(self.motion_duration)
        
        # Get and analyze results
        motion_data = self.motion_detector.stop_detection()
        audio_stress_detected, audio_stress_level, audio_description = self.detect_stress(audio_data, sample_rate)
        motion_stress_level, motion_description = self.motion_detector.analyze_stress()
        
        # Analyze combined stress levels
        is_panic, is_high_stress, combined_level, combined_description = self.analyze_combined_stress(
            audio_stress_detected, audio_stress_level, motion_stress_level
        )
        
        # Store results
        self.results.update({
            'audio_stress_detected': audio_stress_detected,
            'motion_stress_detected': motion_stress_level > self.motion_detector.stress_threshold,
            'audio_stress_level': audio_stress_level,
            'motion_stress_level': motion_stress_level,
            'combined_stress_level': combined_level,
            'panic_detected': is_panic
        })
        
        # Print results
        print("\n=== ANALYSIS RESULTS ===")
        print(f"Voice Analysis: {audio_description}")
        print(f"Movement Analysis: {motion_description}")
        print(f"Overall State: {combined_description}")
        print(f"Combined Stress Level: {combined_level:.1%}")
        
        # Only take action for panic or high stress
        if is_panic:
            print("\n🚨 PANIC DETECTED! 🚨")
            print("Extreme stress indicators observed in both voice and movements.")
            self._handle_stress_detection(True)
        elif is_high_stress:
            print("\n⚠️ HIGH STRESS DETECTED! ⚠️")
            print("Significant stress indicators observed.")
            self._handle_stress_detection(False)
        else:
            print("\n✅ No significant stress or panic detected.")
            print("Your stress levels appear to be within normal ranges.")
        
        # Clean up
        if os.path.exists("temp_recording.wav"):
            os.remove("temp_recording.wav")
        
        return self.results
    
    def _handle_stress_detection(self, is_panic):
        """Handle the detection of high stress or panic."""
        print("\nFetching your location to find nearby relaxation spots...")
        location_data = get_location_by_geocoder()
        self.results['location_data'] = location_data
        
        if location_data:
            print("Location data retrieved successfully!")
            if open_google_maps(location_data):
                if is_panic:
                    print("Opening Google Maps to help you find immediate assistance...")
                else:
                    print("Opening Google Maps to help you find relaxation spots...")
            else:
                print("Failed to open Google Maps.")
        else:
            print("Failed to retrieve location data. Cannot open Google Maps.")
    
    def release(self):
        """Release resources."""
        if hasattr(self, 'motion_detector'):
            self.motion_detector.release()

    def record_audio(self, duration=5, sample_rate=22050):
        """Record audio from microphone."""
        print(f"\nRecording audio for {duration} seconds...")
        print("Please speak now...")
        
        # Create a queue to store the audio data
        q = queue.Queue()
        
        def callback(indata, frames, time, status):
            """Callback function to store audio data in queue."""
            if status:
                print(f"Status: {status}")
            # Increase gain by multiplying the input data
            indata = indata * 2.0  # Double the input gain
            q.put(indata.copy())
        
        # Start recording with higher gain
        with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback):
            sd.sleep(int(duration * 1000))
        
        # Get the recorded data
        data = []
        while not q.empty():
            data.append(q.get())
        
        # Convert to numpy array
        audio_data = np.concatenate(data, axis=0)
        return audio_data, sample_rate

    def detect_stress(self, audio_data, sample_rate):
        """
        Detect stress in the audio data based on extracted features.
        Returns a tuple of (is_stressed, stress_level, description)
        where stress_level is 0.0 (calm) to 1.0 (panic)
        """
        try:
            features = self.extract_features(audio_data, sample_rate)
            if features is None:
                return False, 0.0, "No audio features detected"
            
            # Print feature values for debugging
            print("\nExtracted Audio Features:")
            print(f"Pitch Mean: {features['pitch_mean']:.2f}")
            print(f"Energy Mean: {features['energy_mean']:.2f}")
            print(f"Zero Crossing Rate: {features['zcr_mean']:.2f}")
            print(f"Spectral Centroid: {features['spectral_centroid_mean']:.2f}")
            
            # Calculate stress level for each feature (0.0 to 1.0)
            stress_levels = []
            
            # Pitch stress (normal range: 100-150)
            pitch_stress = min(1.0, max(0.0, (features['pitch_mean'] - 130) / 50))
            stress_levels.append(pitch_stress)
            if pitch_stress > 0.5:
                print("✓ Elevated pitch detected")
            
            # Energy stress (normal range: 0.3-0.5)
            energy_stress = min(1.0, max(0.0, (features['energy_mean'] - 0.5) / 0.5))
            stress_levels.append(energy_stress)
            if energy_stress > 0.5:
                print("✓ Elevated energy detected")
            
            # Zero Crossing Rate stress (normal range: 0.05-0.08)
            zcr_stress = min(1.0, max(0.0, (features['zcr_mean'] - 0.08) / 0.12))
            stress_levels.append(zcr_stress)
            if zcr_stress > 0.5:
                print("✓ Elevated zero crossing rate detected")
            
            # Spectral Centroid stress (normal range: 1500-1800)
            spectral_stress = min(1.0, max(0.0, (features['spectral_centroid_mean'] - 1800) / 700))
            stress_levels.append(spectral_stress)
            if spectral_stress > 0.5:
                print("✓ Elevated spectral centroid detected")
            
            # Calculate overall stress level (average of all stress levels)
            overall_stress = sum(stress_levels) / len(stress_levels)
            
            # Determine stress state
            is_stressed = overall_stress > 0.3  # Lower threshold for stress detection
            description = "Calm" if overall_stress < 0.3 else \
                         "Mild Stress" if overall_stress < 0.5 else \
                         "Moderate Stress" if overall_stress < 0.7 else \
                         "High Stress" if overall_stress < 0.9 else "Panic"
            
            print(f"\nOverall Stress Level: {overall_stress:.2%}")
            print(f"Voice State: {description}")
            
            return is_stressed, overall_stress, description
            
        except Exception as e:
            print(f"Error in stress detection: {str(e)}")
            return False, 0.0, "Error in stress detection"

def main():
    """Main function to run the integrated stress detector."""
    parser = argparse.ArgumentParser(description='Integrated Stress Detector')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0 for front camera)')
    parser.add_argument('--duration', type=int, default=5, help='Recording duration in seconds (default: 5)')
    parser.add_argument('--record', action='store_true', help='Record video during motion detection')
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        print("Installing required dependencies...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
            print("Dependencies installed successfully.")
        except Exception as e:
            print(f"Error installing dependencies: {str(e)}")
            sys.exit(1)
    
    try:
        # Initialize detector
        detector = IntegratedStressDetector(
            camera_id=args.camera,
            audio_duration=args.duration,
            
            motion_duration=args.duration
        )
        
        # Run detection
        detector.run_detection(record_video=args.record)
        
        # Ask if user wants to continue
        choice = input("\nDo you want to run another detection? (y/n): ")
        if choice.lower() == 'y':
            detector.run_detection(record_video=args.record)
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Clean up
        if 'detector' in locals():
            detector.release()
        
        print("\nThank you for using the Integrated Stress Detection Application!")

if __name__ == "__main__":
    main() 