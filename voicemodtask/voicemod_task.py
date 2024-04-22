from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

def voicemod_test_process(white_noise_original, challenge_audio):
    try: 
        # Load audio files
        audio1 = AudioSegment.from_wav(white_noise_original)
        audio2 = AudioSegment.from_wav(challenge_audio)

        # Invert the phase on the white noise file
        inverted_phase_whiteNoise = audio1.invert_phase()

        # Mix the audio files together
        bounced_mix = inverted_phase_whiteNoise.overlay(audio2) 

        # Reverse the bounced audio file
        final_mix = bounced_mix.reverse()

        # Export new audio file
        final_mix.export("processed_audio1.wav", format="wav")

        # Load the output audio file
        output_audio_file = "processed_audio1.wav"
        y, sr = librosa.load(output_audio_file)

        # Compute the spectrogram
        D = librosa.stft(y)

        # Convert to dB scale
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        # Display the spectrogram
        plt.figure(figsize=(12, 2))
        librosa.display.specshow(D_db, x_axis='time', y_axis='log')
        plt.title('Voicemod Audio Spectogram')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.show()

    except Exception as error:
        print("An error occoured:", error)

if __name__ == "__main__":
    # Paths to your audio files
    white_noise_original = "C:/Users/kiera/Downloads/Voicemod_Test/WhiteNoiseMono.wav"
    challenge_audio = "C:/Users/kiera/Downloads/Voicemod_Test/Challenge.wav"

    # Call the function to manipulate audio files
    voicemod_test_process(white_noise_original, challenge_audio)