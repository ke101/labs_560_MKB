import fitz
import sys
import easyocr
from pdf2image import convert_from_path

folder = sys.argv[1]

doc = fitz.open(folder)
txt = folder[:-4]+".txt"


# Extract all text
doc = fitz.open(folder)
all_text = ""

for page in doc:
    all_text += page.get_text()

doc.close()

# in the case, if the PDF is image-based, convert back to image and use OCR
if not all_text.strip():
    print("Detected image-based PDF. Switching to EasyOCR...")

    reader = easyocr.Reader(['en'])
    images = convert_from_path(pdf_path)

    for img in images:
        results = reader.readtext(img)
        for _, text, _ in results:
            all_text += text + "\n"

else:
    print("Detected text-based PDF. Used PyMuPDF.")
# Print first 10 lines
lines = all_text.splitlines()
for line in lines[:10]:
    print(line)

# Save all text to txt file
with open(txt, "w", encoding="utf-8") as f:
    f.write(all_text)