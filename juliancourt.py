import os
import io
import json
import subprocess
#subprocess.call("/home/saul/emails")
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Doc, Span, Token
import spacy
from gensim.models import Word2Vec
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
#import QML_application.py



class Document:
    # Class attributes
    resource_manager = PDFResourceManager()
    file_handle = io.StringIO()
    converter = TextConverter(resource_manager, file_handle)
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    #nlp = spacy.load("en_core_web_sm")
    nlp = spacy.load("en_core_web_lg")
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
            
            text = self.file_handle.getvalue() # whole document in text
            list.append(text)
        #list.to_csv('pdftotext.csv')
        self.converter.close()
        self.file_handle.close()
        #print(text)
        self.__courtAnalysis(text)
        
    def __courtAnalysis(self, text):
        print(type(text))
        # Add law jarjon and terms to stop words
        customize_stop_words = [
        'the', 'to', " \x0c", ' ', 'Mr.', 'Dr.', 'v', 'of', 'case', 'section', 'defence',
        'trial', 'evidence', 'law', 'court', 'Court', 'criminal', 'Act', 'Article', 'UK',
        '“'
        ]
        for w in customize_stop_words:
            self.nlp.vocab[w].is_stop = True
            
        customize_non_punct = [
        '.'
        ]
        for w in customize_non_punct:
            self.nlp.vocab[w].is_punct = False
        
        doc = self.nlp(text)

        #remove stop wods 
        cleanDoc = [t.text for t in doc if t.is_stop != True and t.whitespace_ != True and t.text.isspace() != True and t.is_punct != True]
        #cleanDoc = [t.text for t in doc if t.is_stop != True and t.whitespace_ != True and t.text.isspace() != True]
        #print("Size :", len(cleanDoc))
        
        # convert List to String
        listToStr = ' '.join([str(elem) for elem in cleanDoc]) 
        print(listToStr)
        cleanDoc = self.nlp(listToStr)
        
        # convert list ot nlp doc
        #cleanDoc = Doc(self.nlp.vocab, words=cleanDoc)
        # Tokens of the document
        #tokens = [t.text for t in doc if t.is_stop != True and t.is_punct != True]
        #self.__wordSimilarity(cleanDoc)
        #nouns = [t.lemma_ for t in doc if t.is_stop != True and t.is_punct != True and t.pos_ =="NOUN"]
        #verbs = [t.lemma_ for t in doc if t.is_stop != True and t.is_punct != True and t.pos_ =="VERB"]
        nouns = [t.lemma_ for t in cleanDoc if t.pos_ == "NOUN"]
        verbs = [t.lemma_ for t in cleanDoc if t.pos_ =="VERB"]
        adjectives = [t.lemma_ for t in cleanDoc if t.pos_ == "ADJ"]
        others = [t.lemma_ for t in cleanDoc if t.pos_ != "VERB" and t.pos_ != "NOUN" and t.pos_ != "ADJ" and t.pos_ != "NUM"]
        
        #self.__verbAnalysis(verbs)
        #self.__nounAnalysis(nouns)
        #self.__adjectiveAnalysis(adjectives)
        #self.__otherAnalysis(others)
        #self.__verbSimilarity(verbs)

        #self.__wordAnalysis(tokens, nouns, verbs, adjectives, cleanDoc.ents)
        #print("Tokens", [t.text for t in cleanDoc],'\n')
        # Tags in the document 
        #print("Tags", [(t.text, t.tag_, t.pos_) for t in doc],'\n\n')
        
        # Analyze syntax
        #print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
        #print("Verbs:", [t.lemma_ for t in doc if t.pos_ == "VERB"])
        # Find named entities, phrases and concepts
        #for entity in doc.ents:
            #print(entity.text, entity.label_)
            
    # Get Bag of Words (BoW) of top 10 words
    def __verbAnalysis(self, verbs):
        verb_freq = Counter(verbs)
        common_verbs = verb_freq.most_common(10)
        #print("Common Verbs ", common_verbs)
        self.__verbSimilarity(common_verbs, verbs)
        #self.__converFiletoJSON(common_verbs)
        
    def __nounAnalysis(self, nouns):
        noun_freq = Counter(nouns)
        common_nouns = noun_freq.most_common(10)
        print("Common Nouns ", common_nouns)
    
    def __adjectiveAnalysis(self, adjectives):
        adj_freq = Counter(adjectives)
        common_adjs = adj_freq.most_common(10)
        print("Common Adjectives ", common_adjs)
    
    def __otherAnalysis(self, others):
        oth_freq = Counter(others)
        common_oths = oth_freq.most_common(10)
        print("Common other ", common_oths)
    
    def __converFiletoJSON(self, file):
        #print("File ", file[1][0])
        data = {}
        for value in file:
            print(value)
            data[value[0]]= value[1]
        print(data)

     
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
    
    def __wordSimilarity(self, verbs, document):

        for token1 in verbs:
            for token2 in document:
                if token1.similarity(token2) > 0.9:
                    print(token1.text, token2.text, token1.similarity(token2))

    def __verbSimilarity(self, verbs, document):
    
        toplistToStr = []
        for value in verbs:
            print(value[0])
            toplistToStr.append(value[0])
            
        toplistToStr = ' '.join([str(elem) for elem in toplistToStr]) 
        topcleanVerbs = self.nlp(toplistToStr)
        print(topcleanVerbs)
        verblistToStr = ' '.join([str(elem) for elem in document]) 
        cleanVerbs = self.nlp(verblistToStr)
        #print(cleanVerbs)
        # get other words strongly associated with the top 10. Then plot them
        for token1 in topcleanVerbs:
            for token2 in cleanVerbs:
                if token1.similarity(token2) > 0.5 and token1.similarity(token2) < 1:
                    print(token1.text, token2.text, token1.similarity(token2))
        
    def getcbow(dataset):

        sentences = []
        vectorised_codes = []
    

        #bugs = pd.read_csv('bug-metrics.csv', sep= ',')
        #print(bugs.columns)

        ast = [row.split('::') for row in dataset['classname']]
        #print('ASTs ', ast[:2])
        #the input to the cbow is list of list of each line
        #size of the word vector of a given token must be equal to embedding_dim of the LSTM model
        cbowmodel = Word2Vec(ast, min_count=1, size= embedding_dims, workers=3, window=3, sg=0)
        #print(ast[:2])
        print (' CBOW model ', cbowmodel)
    
        #Test cbow model
        print("Test CBOW on the data")
        print(cbowmodel['eclipse'])
    
        classes = dataset['classname']

        for codes in classes:

            linecode = []
            tokens = codes.split('::')
            #print(tokens)
            sentences.append(tokens)
            for token in tokens:
                try:
                    #print("Token ", token)
                    #linecode.append(token)
                    #print("Word Vector ", len(cbowmodel[token]))
                    linecode.append(cbowmodel[token])
                except KeyError:
                    pass
        vectorised_codes. append(linecode)
    #print(len(linecode))
    #print(linecode)


    #print('Line codes ', linecode)
    #print('Vectorised Codes ', vectorised_codes[0])
    #print('Vectorised Codes ', len(vectorised_codes))
    #print(f'Sentences: {sentences}')

        return vectorised_codes

if __name__ == '__main__':
    courtdoco = Document("usaassangejudgement.pdf")
