import os
import requests
import zipfile
from tqdm import tqdm
import shutil

# --- Konfigurasi Dataset ---
# DIBCO datasets yang tersedia dan terverifikasi
DATASETS = {
    # H-DIBCO 2016 (Handwritten Document Image Binarization Contest)
    "HDIBCO2016_Original": {
        "url": "https://vc.ee.duth.gr/h-dibco2016/benchmark/DIBCO2016_dataset-original.zip",
        "source": "University of Thessaly",
        "size_est": "~8.6MB",
        "verified": True,
        "note": "Original handwritten document images"
    },
    "HDIBCO2016_GroundTruth": {
        "url": "https://vc.ee.duth.gr/h-dibco2016/benchmark/DIBCO2016_dataset-GT.zip",
        "source": "University of Thessaly",
        "size_est": "~235KB",
        "verified": True,
        "note": "Ground truth binary images"
    },
    "HDIBCO2016_Metrics": {
        "url": "https://vc.ee.duth.gr/h-dibco2016/benchmark/DIBCO_metrics.zip",
        "source": "University of Thessaly",
        "size_est": "~5.1MB",
        "verified": True,
        "note": "Evaluation metrics and tools"
    },
    "HDIBCO2016_Weights": {
        "url": "https://vc.ee.duth.gr/h-dibco2016/benchmark/BinEvalWeights.zip",
        "source": "University of Thessaly",
        "size_est": "~2.9MB",
        "verified": True,
        "note": "Binary evaluation weights"
    },

    # Coba DIBCO 2017 (kemungkinan tersedia)
    "DIBCO2017_Temp": {
        "url": "https://vc.ee.duth.gr/dibco2017/DIBCO2017_dataset.zip",
        "source": "University of Thessaly",
        "size_est": "~50MB",
        "verified": False,
        "note": "May not be available yet - need to check"
    }
}

# Folder utama untuk menyimpan semua dataset
DOWNLOAD_DIR = "dibco_datasets"

def validate_file(filepath: str) -> tuple[bool, str]:
    """
    Validasi apakah file adalah ZIP atau gambar yang valid.
    Returns: (is_valid, file_type)
    """
    try:
        # Check if ZIP file
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.testzip()
        return True, "zip"
    except zipfile.BadZipFile:
        # Check if image file
        try:
            from PIL import Image
            with Image.open(filepath) as img:
                img.verify()
            return True, "image"
        except Exception:
            return False, "unknown"
    except Exception:
        return False, "unknown"

def download_file(url: str, destination: str, dataset_name: str = "") -> bool:
    """
    Mengunduh file dari URL dengan progress bar dan validasi.
    """
    print(f"📥 Downloading {dataset_name} from {url}")

    try:
        # Set headers untuk mengidentifikasi sebagai browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, stream=True, headers=headers, timeout=30)
        response.raise_for_status()

        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if 'html' in content_type:
            print(f"❌ Error: URL returned HTML page, not a file. Content-Type: {content_type}")
            return False

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB

        with open(destination, 'wb') as f, tqdm(
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {os.path.basename(destination)}"
        ) as bar:
            for data in response.iter_content(block_size):
                if data:  # filter out keep-alive new chunks
                    bar.update(len(data))
                    f.write(data)

        # Validate download
        if total_size != 0 and bar.n != total_size:
            print(f"⚠️  Warning: Downloaded size ({bar.n}) differs from expected ({total_size})")

        # Validate file (ZIP or image)
        is_valid, file_type = validate_file(destination)
        if not is_valid:
            print(f"❌ Error: Downloaded file is not a valid ZIP or image file")
            os.remove(destination)  # Hapus file yang tidak valid
            return False

        print(f"✅ File type detected: {file_type}")

        print(f"✅ Successfully downloaded and validated {os.path.basename(destination)}")
        return True

    except requests.exceptions.Timeout:
        print(f"❌ Error: Download timeout for {url}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading {url}: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def unzip_file(zip_path: str, extract_to: str):
    """
    Mengekstrak file zip ke folder tujuan.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ Successfully unzipped {os.path.basename(zip_path)}")
        return True
    except zipfile.BadZipFile:
        print(f"❌ Error: {os.path.basename(zip_path)} is not a valid zip file.")
        return False
    except Exception as e:
        print(f"❌ Error unzipping {zip_path}: {e}")
        return False

def main():
    """
    Fungsi utama untuk mengunduh dan mengekstrak dataset.
    """
    # Buat folder utama jika belum ada
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    print(f"🗂️  All datasets will be saved in the '{DOWNLOAD_DIR}' folder.\n")

    successful_downloads = []
    failed_downloads = []

    for name, config in DATASETS.items():
        print(f"--- Processing {name} ---")
        print(f"📍 Source: {config['source']}")
        print(f"📏 Estimated size: {config.get('size_est', 'Unknown')}")

        # Skip unverified datasets unless explicitly requested
        if not config.get('verified', True):
            print(f"⚠️  Skipping {name} - not verified")
            print("-" * (len(name) + 20) + "\n")
            continue

        url = config['url']

        # Path untuk file zip yang akan diunduh
        zip_filename = f"{name}.zip"
        zip_filepath = os.path.join(DOWNLOAD_DIR, zip_filename)

        # Nama folder hasil ekstrak
        extract_path = os.path.join(DOWNLOAD_DIR, name)

        # Cek apakah folder hasil ekstrak sudah ada
        if os.path.exists(extract_path):
            print(f"⏭️  Dataset '{name}' already exists at '{extract_path}'. Skipping.")
            print("-" * (len(name) + 20) + "\n")
            successful_downloads.append(name)
            continue

        # Unduh file
        success = download_file(url, zip_filepath, name)
        if not success:
            failed_downloads.append(name)
            print("-" * (len(name) + 20) + "\n")
            continue

        # Process file (extract if ZIP, or copy if image)
        is_valid, file_type = validate_file(zip_filepath)

        if file_type == "zip":
            success = unzip_file(zip_filepath, DOWNLOAD_DIR)
        elif file_type == "image":
            success = True  # Image files don't need extraction
            print(f"📁 Image file saved directly: {zip_filepath}")
        else:
            success = False

        if success:
            if file_type == "zip":
                # Rename folder jika nama ekstrak berbeda
                extracted_folders = [f for f in os.listdir(DOWNLOAD_DIR)
                                  if os.path.isdir(os.path.join(DOWNLOAD_DIR, f))
                                  and f.startswith(zip_filename.replace('.zip', ''))]

                if extracted_folders:
                    old_path = os.path.join(DOWNLOAD_DIR, extracted_folders[0])
                    if old_path != extract_path:
                        shutil.move(old_path, extract_path)
                        print(f"📁 Renamed folder to: {name}")

                # Hapus file zip setelah berhasil diekstrak
                try:
                    os.remove(zip_filepath)
                    print(f"🗑️  Removed zip file: {zip_filename}")
                except OSError as e:
                    print(f"❌ Error removing zip file: {e}")

                # Tampilkan info dataset yang berhasil diunduh
                dataset_files = []
                for _, _, files in os.walk(extract_path):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                            dataset_files.append(file)

                print(f"📊 Dataset info: Found {len(dataset_files)} image files")

            elif file_type == "image":
                # Rename image file untuk lebih deskriptif
                new_filename = f"{name}.png"
                new_filepath = os.path.join(DOWNLOAD_DIR, new_filename)
                try:
                    shutil.move(zip_filepath, new_filepath)
                    print(f"📁 Renamed image to: {new_filename}")
                    print(f"📊 Dataset info: Found 1 image file")
                except OSError as e:
                    print(f"❌ Error renaming file: {e}")

            successful_downloads.append(name)

        else:
            failed_downloads.append(name)

        print("-" * (len(name) + 20) + "\n")

    # Summary
    print("\n" + "="*50)
    print("📋 DOWNLOAD SUMMARY")
    print("="*50)

    if successful_downloads:
        print(f"✅ Successfully downloaded: {', '.join(successful_downloads)}")
        print(f"📁 Location: {os.path.abspath(DOWNLOAD_DIR)}")

        # Generate usage instructions
        print(f"\n📖 USAGE INSTRUCTIONS:")
        for dataset_name in successful_downloads:
            dataset_path = os.path.join(DOWNLOAD_DIR, dataset_name)
            print(f"   {dataset_name}: {dataset_path}")

    if failed_downloads:
        print(f"❌ Failed to download: {', '.join(failed_downloads)}")
        print(f"\n💡 TIPS for failed downloads:")
        print(f"   1. Check your internet connection")
        print(f"   2. Try again later (servers may be temporarily unavailable)")
        print(f"   3. Some datasets may require manual download from official websites")

    print(f"\n🎉 Process finished! {len(successful_downloads)} datasets ready for use.")

if __name__ == "__main__":
    main()