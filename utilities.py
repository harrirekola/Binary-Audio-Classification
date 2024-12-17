import os
from pydub import AudioSegment

def rename_files(directory, prefix="car_idle"):
    """
    Renames all files in the specified directory without changing their extensions.
    
    Parameters:
        directory (str): Path to the directory containing files.
        prefix (str): Prefix for the renamed files (default is "file").
    """
    # Check if the provided path is a directory
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return
    
    # List all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Sort files for consistent order
    files.sort()

    # Rename files while keeping their extensions
    for index, file_name in enumerate(files, start=1):
        file_path = os.path.join(directory, file_name)
        file_extension = os.path.splitext(file_name)[1]  # Get file extension
        new_name = f"{prefix}_{index}{file_extension}"   # Create new name
        new_path = os.path.join(directory, new_name)
        
        try:
            os.rename(file_path, new_path)
            print(f'Renamed: "{file_name}" -> "{new_name}"')
        except Exception as e:
            print(f"Error renaming {file_name}: {e}")

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
