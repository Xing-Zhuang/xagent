 
import easyocr


 
def ocr_read_text(image_path: str):
    reader = easyocr.Reader(['ch_sim','en'])  
    res = reader.readtext(image_path, detail = 0)
    return res
docs_ocr_read_text="""
This is a ocr tool for recognizing text in images.

Args:
    image_path(str):The image path.
"""