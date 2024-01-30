import pdfminer
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
from spacy.matcher import PhraseMatcher, Matcher
import spacy
import os
import io
import csv

def extract_text_from_pdf(pdf_path):
	list = []
	resource_manager = PDFResourceManager()
	laparams = pdfminer.layout.LAParams()
	setattr(laparams, 'all_texts', True)
        
	fake_file_handle = io.StringIO()
	converter = TextConverter(resource_manager, fake_file_handle, laparams=laparams)
	page_interpreter = PDFPageInterpreter(resource_manager, converter)
	with open(pdf_path, 'rb') as fh:
         for page in PDFPage.get_pages(fh, 
                                       caching=True,
                                       check_extractable=True):
										   page_interpreter.process_page(page)
         text = fake_file_handle.getvalue() # whole document in text
            #print('Text ', text)       
         list.append(text)
         
	csv_writer = csv.writer(sys.stdout, delimiter='\t')
	with open(r'/home/saul/Desktop/generative-AI/document_analysis/court_document_topics.txt', 'w') as fp:
		for item in text:
			# write each item on a new line
            #item = item + ' '
			fp.write("%s\n" % item)
            #fp.write("%s" % item)

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
    os.chdir('/home/saul/Desktop/generative-AI/document_analysis')
    pdf_path = "Redacted Indictment Translation.pdf"  # Replace with the new PDF file
    text = extract_text_from_pdf(pdf_path)

    topics = perform_topic_analysis(text)

    print("Extracted Topics:")
    for i, topic in enumerate(topics, start=1):
        print(f"{i}. {topic}")

if __name__ == "__main__":
    main()
