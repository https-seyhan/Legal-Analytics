import os
import io
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Doc, Span, Token
import spacy

class Document:
    # Class attributes
    resource_manager = PDFResourceManager()
    file_handle = io.StringIO()
    converter = TextConverter(resource_manager, file_handle)
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    nlp = spacy.load("en_core_web_sm")
    os.chdir('/home/saul/Business')
    
    def __init__(self, fileName):
        print("constructor called")
        print ("Class Attributes ", self.resource_manager, self.file_handle, self.converter)
        self.__convertToText(fileName)
        
    def __convertToText(self, fileName):
        list = []
        print ("file Name ", fileName)
        with open(fileName, 'rb') as fh:
            for page in PDFPage.get_pages(fh, 
                                        caching=True,
                                        check_extractable=True):
                self.page_interpreter.process_page(page)
            
            text = self.file_handle.getvalue()
            list.append(text)
        #list.to_csv('pdftotext.csv')
        #print("List ", list)
        self.converter.close()
        self.file_handle.close()
        self.__courtAnalysis(text)
        
    def __courtAnalysis(self, text):
        #print(text)
        
        doc = self.nlp(text)
        #remove stop wods 
        cleanDoc = [t.text for t in doc if not t.is_stop]
        print("Size :", len(cleanDoc))
        #print(doc)
        # convert list ot nlp doc
        cleanDoc = Doc(self.nlp.vocab, words=cleanDoc)
        # Tokens of the document
        #print("Tokens", [t.text for t in cleanDoc],'\n')
        # Tags in the document 
        print("Tags", [(t.text, t.tag_, t.pos_) for t in cleanDoc],'\n\n')
        
        
if __name__ == '__main__':
    courtdoco = Document("usaassangejudgement.pdf")
