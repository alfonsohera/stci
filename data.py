
import os
import requests
import shutil

from zipfile import ZipFile
from google.colab import drive
from pydub import AudioSegment

drive.mount('/content/drive')
drive_path = '/content/drive/MyDrive/ModelCheckpoints'
os.makedirs(drive_path, exist_ok=True)


def DownloadAndExtract():
    urls = [
        "https://ars.els-cdn.com/content/image/1-s2.0-S0885230821001340-mmc2.zip",
        "https://ars.els-cdn.com/content/image/1-s2.0-S0885230821001340-mmc3.zip",
        "https://ars.els-cdn.com/content/image/1-s2.0-S0885230821001340-mmc4.zip",
        "https://ars.els-cdn.com/content/image/1-s2.0-S0885230821001340-mmc5.zip",
        "https://ars.els-cdn.com/content/image/1-s2.0-S0885230821001340-mmc6.zip",
        "https://ars.els-cdn.com/content/image/1-s2.0-S0885230821001340-mmc7.zip",
        "https://ars.els-cdn.com/content/image/1-s2.0-S0885230821001340-mmc8.zip",
        "https://ars.els-cdn.com/content/image/1-s2.0-S0885230821001340-mmc9.zip",
        "https://ars.els-cdn.com/content/image/1-s2.0-S0885230821001340-mmc10.zip",
        "https://ars.els-cdn.com/content/image/1-s2.0-S0885230821001340-mmc11.zip"
    ]

    # Create root directories for your final data
    os.makedirs("Healthy", exist_ok=True)
    os.makedirs("MCI", exist_ok=True)
    os.makedirs("AD", exist_ok=True)
    # Create a temporary folder for extracted files
    temp_folder = "tmp_extracted"
    os.makedirs(temp_folder, exist_ok=True)
    # Download and extract each zip file
    for i, url in enumerate(urls):
        zip_filename = f"downloaded_{i}.zip"   # A local name to store the downloaded file
        # Download the file
        print(f"Downloading from {url}...")
        response = requests.get(url)
        with open(zip_filename, "wb") as f:
            f.write(response.content)
        print(f"Saved {zip_filename}")
        # Extract all contents into the temp_folder
        print(f"Extracting {zip_filename}...")
        with ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(temp_folder)
        # Delete the ZIP file after extraction to save space
        os.remove(zip_filename)
    temp_folder = "tmp_extracted"
    # Move files to their destinations, convert from mp3 to wav if needed
    for root, dirs, files in os.walk(temp_folder):
        for filename in files:
            full_path = os.path.join(root, filename)
            # Convert MP3 to WAV if needed
            if filename.endswith(".mp3"):
                wav_filename = filename.replace(".mp3", ".wav")
                wav_path = os.path.join(root, wav_filename)
                # Convert MP3 to WAV
                audio = AudioSegment.from_mp3(full_path)
                audio.export(wav_path, format="wav")
                # Remove the original MP3
                os.remove(full_path)
                # Update full_path to the new WAV file
                full_path = wav_path
                filename = wav_filename
            # Move to corresponding folder
            if filename.startswith("AD"):
                shutil.move(full_path, os.path.join("AD", filename))
            elif filename.startswith("MCI"):
                shutil.move(full_path, os.path.join("MCI", filename))
            elif filename.startswith("HC"):
                shutil.move(full_path, os.path.join("Healthy", filename))
            else:
                print(f"File '{filename}' doesn't match AD/MCI/HC. Skipping or placing it elsewhere.")
    print("MP3 conversion and file moving completed.")
    # Delete temporary folder
    shutil.rmtree(temp_folder, ignore_errors=True)
    print("Temporary folder removed.")
