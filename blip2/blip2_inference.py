from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import warnings

warnings.filterwarnings('ignore')
def generate_features(image_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    raw_image = Image.open(image_path).convert('RGB')
    text = f""
    # print(image_path)
    inputs = processor(raw_image, text, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

if __name__ == "__main__":
    image_path = 'test.jpeg'
    print("Code started...")
    generate_features(image_path)
