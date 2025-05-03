# title: vibe coding test using pytesseract
# date: 2025-03-01
# install: 
#   1) Tesseract tool with language pack(Korean, Math). https://github.com/UB-Mannheim/tesseract/wiki
#   2) set the tesseract.exe path in the environment variable or set it in the code below.
#   3) test 'where tesseract.exe' in terminal
#   4) pip install pytesseract pdf2image pandas openpyxl pillow
#
from PIL import Image, ImageDraw
import pytesseract # check https://pypi.org/project/pytesseract/, https://github.com/tesseract-ocr/tesseract
import pandas as pd
import os

# vibe coding prompt: read image files in './input' folder, using pytesseract, convert them to the pandas dataframe which consists of no, filename, description column and save 'output.xlsx' excel file. 

# Define the input folder and output file
input_folder = './input'
output_folder = './output'
output_file = 'output.xlsx'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Initialize an empty list to store the data
data = []

# Iterate through files in the input folder
for idx, filename in enumerate(os.listdir(input_folder)):
    file_path = os.path.join(input_folder, filename)
    
    # Check if the file is an image or a PDF
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # Process image files
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        
        # Get bounding box data
        boxes = pytesseract.image_to_boxes(img)
        
        # Draw bounding boxes on the image
        draw = ImageDraw.Draw(img)
        img_width, img_height = img.size
        for box in boxes.splitlines():
            b = box.split()
            x1, y1, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
            y1, y2 = img_height - y2, img_height - y1
            draw.rectangle([x1, y1, x2, y2], outline="red", width=1)
        
        # Save the image with bounding boxes to the output folder
        output_image_path = os.path.join(output_folder, f"bbox_{filename}")
        img.save(output_image_path)
    else:
        # Skip unsupported file types
        continue
    
    # Append the data to the list
    data.append({'no': idx + 1, 'filename': filename, 'description': text.strip()})

# Convert the list to a pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to an Excel file
df.to_excel(output_file, index=False)
print(f"Data successfully saved to {output_file}")
print(f"Bounding box images saved to {output_folder}")