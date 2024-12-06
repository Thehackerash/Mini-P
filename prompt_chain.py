import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

# Configure the Gemini API (replace with your actual API key))
genai.configure(api_key=os.getenv("api_key"))

# Initialize the Gemini model
gemini_model = genai.GenerativeModel('gemini-pro')

# Multi-Level Prompts
def object_level_prompt(task):
    return f"""What common objects in daily life can be used as a vehicle for the human to {task}? 
    Please list the twenty most suitable objects."""

def affordance_level_prompt(task, objects):
    return f"""For the task '{task}', consider the objects: {objects}. 
    For each object, let's think about the rationales for why they afford the task 
    from the perspective of visual features."""

def visual_feature_prompt(task, rationales):
    return f"""For the task '{task}', based on these rationales: {rationales}, 
    summarize the corresponding visual features and material (like plastic, metal) for each object. in form of a single string"""

# Multi-Level Chain-of-Thought Process
def multi_level_cot(task):
    # Step 1: Object-Level Reasoning
    object_response = gemini_model.generate_content(object_level_prompt(task))
    objects = object_response.text
    print("\n--- Objects Identified ---\n", objects)
    
    # Step 2: Affordance-Level Reasoning
    affordance_response = gemini_model.generate_content(
        affordance_level_prompt(task, objects)
    )
    rationales = affordance_response.text
    print("\n--- Rationales Generated ---\n", rationales)
    
    # Step 3: Visual Feature Summarization
    visual_feature_response = gemini_model.generate_content(
        visual_feature_prompt(task, rationales)
    )
    visual_features = visual_feature_response.text
    print("\n--- Visual Features Summarized ---\n", visual_features)
    
    return visual_features

# Example Usage
if __name__ == "__main__":
    task = "open a parcel"
    results = multi_level_cot(task)
    print("\n--- Final Results ---\n", results)