import os
import io
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Doc, Span, Token
import spacy
from collections import Counter

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
        cleanDoc = [t.text for t in doc if t.is_stop != True and t.is_punct != True]
        print("Size :", len(cleanDoc))
        #print(doc)
        # convert list ot nlp doc
        cleanDoc = Doc(self.nlp.vocab, words=cleanDoc)
        # Tokens of the document
        tokens = [t.text for t in doc if t.is_stop != True and t.is_punct != True]
        #nouns = [t.lemma_ for t in doc if t.is_stop != True and t.is_punct != True and t.pos_ =="NOUN"]
        #verbs = [t.lemma_ for t in doc if t.is_stop != True and t.is_punct != True and t.pos_ =="VERB"]
        nouns = [t.lemma_ for t in doc if t.pos_ == "NOUN"]
        verbs = [t.lemma_ for t in doc if t.pos_ == "VERB"]
        adjectives = [t.lemma_ for t in doc if t.pos_ == "ADJ"]
        #other = [t.lemma_ for t in doc if t.pos_ != "VERB" and t.pos_ != "NOUN"]
        #print(verbs)

        self.__wordAnalysis(tokens, nouns, verbs, adjectives, cleanDoc.ents)
        #print("Tokens", [t.text for t in cleanDoc],'\n')
        # Tags in the document 
        #print("Tags", [(t.text, t.tag_, t.pos_) for t in doc],'\n\n')
        
        # Analyze syntax
        #print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
        #print("Verbs:", [t.lemma_ for t in doc if t.pos_ == "VERB"])
        # Find named entities, phrases and concepts
        #for entity in doc.ents:
            #print(entity.text, entity.label_)
    
    def __wordAnalysis(self, tokens, nouns, verbs, adjectives, docents):
        #print(verbs)
        # five most common tokens
        verb_freq = Counter(verbs)
        common_verbs = verb_freq.most_common(50)
        print("Common Verbs ", common_verbs)
        
        noun_freq = Counter(nouns)
        common_nouns = noun_freq.most_common(50)
        print("Common Nouns ", common_nouns)
        
        token_freq = Counter(tokens)
        common_tokens = token_freq.most_common(50)
        print("Common Tokens ", common_tokens)
        
        adj_freq = Counter(adjectives)
        common_adjs = adj_freq.most_common(50)
        print("Common adjectives ", common_adjs)
        
        
        
        
if __name__ == '__main__':
    courtdoco = Document("usaassangejudgement.pdf")
