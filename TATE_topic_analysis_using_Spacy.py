import PyPDF2
import spacy
import os

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfFileReader(file)
        for page_num in range(pdf_reader.numPages):
            page = pdf_reader.getPage(page_num)
            text += page.extractText()
    return text

def perform_topic_analysis(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    # Analyze topics based on spaCy's named entities
    topics = set()
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "LOC"]:
            topics.add(ent.text)

    return topics

def main():
    pdf_path = "Redacted Indictment Translation.pdf"  # Replace with the new PDF file
    text = extract_text_from_pdf(pdf_path)

    topics = perform_topic_analysis(text)

    print("Extracted Topics:")
    for i, topic in enumerate(topics, start=1):
        print(f"{i}. {topic}")

if __name__ == "__main__":
    main()
