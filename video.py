import os
import shutil
import string

# === CONFIGURATION ===
source_folder = r"C:\Users\tanin\OneDrive\Desktop\Emerging Trends\Sign Language\extra"
destination_base = r"C:\Users\tanin\OneDrive\Desktop\Emerging Trends\Sign Language\extra"
file_prefix = "frame_"
file_extension = ".jpg"
starting_letter = input("Letter = ")
start_frame = int(input("Initial Frame Start = "))  # Only once
# ======================

print("🔁 Auto Frame Mover - Auto Folder Name (E-Z), Type 'done' to stop\n")

alphabet = list(string.ascii_uppercase)
folder_index = alphabet.index(starting_letter.upper())

while folder_index < len(alphabet):
    end_input = input(f"Frame End (start={start_frame}) for folder '{alphabet[folder_index]}' = ")
    if end_input.lower() == "done":
        break

    try:
        end_frame = int(end_input)
        folder = alphabet[folder_index]
        folder_index += 1

        dest_folder = os.path.join(destination_base, folder)
        os.makedirs(dest_folder, exist_ok=True)

        moved = 0
        for i in range(start_frame, end_frame + 1):
            filename = f"{file_prefix}{i}{file_extension}"
            src_path = os.path.join(source_folder, filename)
            dst_path = os.path.join(dest_folder, filename)

            if os.path.exists(src_path):
                shutil.move(src_path, dst_path)
                moved += 1
            else:
                print(f"❌ Missing: {filename}")

        print(f"✅ Moved {moved} frames to folder '{folder}'\n")
        start_frame = end_frame + 1

    except ValueError:
        print("⚠️ Invalid input. Please enter a valid number.\n")

print("\n👋 All done!")
