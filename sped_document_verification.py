import os
import openai
from dotenv import load_dotenv
from transformers import pipeline
import gradio as gr

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
huggingface_api_key = os.getenv("HF_TOKEN")

# Hugging Face pipeline for document analysis (example: text classification)
classifier = pipeline("text-classification", model="distilbert-base-uncased", use_auth_token=huggingface_api_key)

# IEP compliance checklist
IEP_CHECKLIST = [
    "Does the IEP include measurable annual goals?",
    "Does the IEP specify the services to be provided?",
    "Is there a statement of the child's present levels of academic achievement?",
    "Does the IEP include a transition plan for students aged 16 or older?",
    "Are the accommodations and modifications clearly outlined?",
]

def analyze_document(document_text):
    """
    Analyze the document text for compliance with the IEP checklist.
    """
    results = []
    for item in IEP_CHECKLIST:
        # Use OpenAI's GPT to evaluate each checklist item
        prompt = f"Review the following document for the item: '{item}'.\n\nDocument:\n{document_text}\n\nDoes the document meet this requirement? Provide a brief explanation."
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            temperature=0.5
        )
        results.append({"checklist_item": item, "response": response.choices[0].text.strip()})
    return results

def classify_document(document_text):
    """
    Classify the document using Hugging Face's text classification pipeline.
    """
    classification = classifier(document_text)
    return classification

def verify_document(document_text):
    """
    Verify the document for IEP compliance and provide classification.
    """
    compliance_results = analyze_document(document_text)
    classification_results = classify_document(document_text)
    return compliance_results, classification_results

def gradio_interface(document_text):
    """
    Gradio interface function to display results.
    """
    compliance_results, classification_results = verify_document(document_text)
    compliance_output = "\n".join(
        [f"{item['checklist_item']}: {item['response']}" for item in compliance_results]
    )
    classification_output = "\n".join(
        [f"Label: {result['label']}, Score: {result['score']:.2f}" for result in classification_results]
    )
    return compliance_output, classification_output

# Gradio UI
interface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(lines=20, placeholder="Paste the IEP document text here..."),
    outputs=[
        gr.Textbox(label="IEP Compliance Results"),
        gr.Textbox(label="Document Classification Results"),
    ],
    title="IEP Document Verification",
    description="Upload an IEP document to verify compliance with the checklist and classify its content."
)

if __name__ == "__main__":
    interface.launch()