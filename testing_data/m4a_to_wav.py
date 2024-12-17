import os
from pydub import AudioSegment

def convert_m4a_to_wav(directory):
    """
    Converts all .m4a files in the specified directory to .wav format.

    Parameters:
        directory (str): Path to the directory containing .m4a files.
    """
    # Check if directory is valid
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
        return

    # Get a list of all .m4a files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.m4a')]

    if not files:
        print("No .m4a files found in the directory.")
        return

    # Process and convert each .m4a file
    for file_name in files:
        m4a_path = os.path.join(directory, file_name)
        wav_name = os.path.splitext(file_name)[0] + ".wav"
        wav_path = os.path.join(directory, wav_name)

        try:
            # Load .m4a file and export as .wav
            audio = AudioSegment.from_file(m4a_path, format="m4a")
            audio.export(wav_path, format="wav")
            print(f"Converted: '{file_name}' -> '{wav_name}'")
        except Exception as e:
            print(f"Error converting '{file_name}': {e}")

if __name__ == "__main__":
    # Prompt user for directory input
    target_directory = input("Enter the directory containing .m4a files: ").strip()
    convert_m4a_to_wav(target_directory)
