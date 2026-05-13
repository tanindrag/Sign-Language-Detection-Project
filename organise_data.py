import os
import shutil
import string


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

source_folder = os.path.join(BASE_DIR, "extra")
destination_base = os.path.join(BASE_DIR, "extra")

file_prefix = "frame_"
file_extension = ".jpg"

print("Auto Frame Mover - Preprocessing Utility")
print("This tool organizes raw frames into alphabet-labeled folders.\n")

if not os.path.exists(source_folder):
    print(f"Error: Source folder '{source_folder}' not found.")
    exit()

try:
    starting_letter = input("Enter starting Letter (e.g., A): ").upper()
    start_frame = int(input("Enter Initial Frame Number to start from: "))
except ValueError:
    print(" Invalid input. Please restart and enter valid data.")
    exit()

alphabet = list(string.ascii_uppercase)

if starting_letter not in alphabet:
    print(" Invalid letter. Please enter A-Z.")
    exit()

folder_index = alphabet.index(starting_letter)

while folder_index < len(alphabet):
    current_label = alphabet[folder_index]
    prompt = f"Enter End Frame for folder '{current_label}' (starting at {start_frame}) or type 'done': "
    end_input = input(prompt)
    
    if end_input.lower() == "done":
        break

    try:
        end_frame = int(end_input)
        dest_folder = os.path.join(destination_base, current_label)
        os.makedirs(dest_folder, exist_ok=True)

        moved = 0
        for i in range(start_frame, end_frame + 1):
            filename = f"{file_prefix}{i}{file_extension}"
            src_path = os.path.join(source_folder, filename)
            dst_path = os.path.join(dest_folder, filename)

            if os.path.exists(src_path):
                shutil.move(src_path, dst_path)
                moved += 1

        print(f" Moved {moved} frames to folder '{current_label}'\n")
        start_frame = end_frame + 1
        folder_index += 1

    except ValueError:
        print(" Invalid input. Please enter a valid number.\n")

print("\n Data organization complete!")
