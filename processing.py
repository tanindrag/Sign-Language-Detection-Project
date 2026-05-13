import os
import string

# Base directory where folders will be created
base_dir = r"C:\Users\tanin\OneDrive\Desktop\Emerging Trends\Sign Language\extra"

# Create folders A to Y (excluding Z)
for letter in string.ascii_uppercase[:-1]:  # A to Y
    folder_path = os.path.join(base_dir, letter)
    os.makedirs(folder_path, exist_ok=True)

print("Folders A to Y created successfully.")
