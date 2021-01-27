from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import pandas as pd
from math import pi
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Doc, Span, Token
import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import operator
import io
import os
os.chdir('/home/saul/Business')
    
class Document:
    # Class attributes
    resource_manager = PDFResourceManager()
    file_handle = io.StringIO()
    converter = TextConverter(resource_manager, file_handle)
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    #nlp = spacy.load("en_core_web_sm")
    nlp = spacy.load("en_core_web_lg")
    tokenizer = Tokenizer(nlp.vocab)
    
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
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
        self.__textAnalysis(text)
        
    def __textAnalysis(self, text):
        print(type(text))
        
        # Add law jargon and terms to stop words
        customize_stop_words = ['a.', 'b.', 'c.', 'i.', 'ii', 'iii', 
        'the', 'to', " \x0c", ' ', 'Mr.', 'Dr.', 'v', 'of', 'case', 'section', 'defence',
        'trial', 'evidence', 'law', 'court', 'Court', 'criminal', 'Act', 'Article', 'UK','extradition', 'offence', 'information',
        '“'
        ]
        for w in customize_stop_words:
            self.nlp.vocab[w].is_stop = True
            
        customize_non_punct = [
        'Ms.'
        ]
        for w in customize_non_punct:
            self.nlp.vocab[w].is_punct = False
        
        doc = self.nlp(text)

        #remove stop wods 
        cleanDoc = [t.text for t in doc if t.is_stop != True and t.whitespace_ != True and t.text.isspace() != True and t.is_punct != True 
        and t.pos != "-PRON-"]
        #cleanDoc = [t.text for t in doc if t.is_stop != True and t.whitespace_ != True and t.text.isspace() != True]
        #print("Size :", len(cleanDoc))
        
        # convert List to String
        listToStr = ' '.join([str(elem) for elem in cleanDoc]) 
        #print(listToStr) # Print clean data
        cleanDoc = self.nlp(listToStr)
        #print(" Clean Doc ", cleanDoc)
        self.__tokenizeDoco(cleanDoc)
        # convert list ot nlp doc
        #cleanDoc = Doc(self.nlp.vocab, words=cleanDoc)
        # Tokens of the document
        #tokens = [t.text for t in doc if t.is_stop != True and t.is_punct != True]
        
        #nouns = [t.lemma_ for t in doc if t.is_stop != True and t.is_punct != True and t.pos_ =="NOUN"]
        #verbs = [t.lemma_ for t in doc if t.is_stop != True and t.is_punct != True and t.pos_ =="VERB"]
        nouns = [t.lemma_ for t in cleanDoc if t.pos_ == "NOUN"]
        verbs = [t.lemma_ for t in cleanDoc if t.pos_ =="VERB"]
        adjectives = [t.lemma_ for t in cleanDoc if t.pos_ == "ADJ"]
        others = [t.lemma_ for t in cleanDoc if t.pos_ != "VERB" and t.pos_ != "NOUN" and t.pos_ != "ADJ" and t.pos_ != "NUM" 
        and t.pos != "-PRON-"]
        #print("Nouns ", nouns)
        #self.__verbAnalysis(verbs)
        #self.__nounAnalysis(nouns)
        #self.__lemmatizerDoco(nouns)
        #self.__lemmatizerDoco(verbs)
        #self.__lemmatizerDoco(adjectives)
        #self.__lemmatizerDoco(others)
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
    def __tokenizeDoco(self, doc):
        vec = CountVectorizer(min_df =0.001, max_df=0.95) # Convert a collection of text documents to a matrix of token counts
        #tokenize = self.tokenizer(text)
        #sentences = [sent.string.strip() for sent in text.sents]
        #print(sentences)
        sents_list = []
        for sent in doc.sents:
            sents_list.append(sent.text)
        #print(sents_list)
        bow_vector = CountVectorizer(tokenizer = sents_list, ngram_range=(1,1))
        #print(bow_vector)
        tfidf_vector = TfidfVectorizer(tokenizer = sents_list)
        #print(tfidf_vector)
        model = vec.fit_transform(sents_list)
        #print("Model ", model)
        print(type(sents_list))
        sentenceCount = 0
        #print(vec.get_feature_names())
        while len(sents_list) > sentenceCount:
            self.__getPurpose(model, sents_list[sentenceCount], sentenceCount)
            sentenceCount += 1
        
    def __getPurpose(self, model, clean_text, sentenceNum):
        # print("Model Values ", len(model[0].todense()))
        print("Index Num ", sentenceNum)
        # get weights of words
        wordweights = model[sentenceNum].data
        #print(" Clean Text ", clean_text)
        words = clean_text.split(" ")
        print("Words ", words)
        sentencepurpose = {}
        #get tfidf vectors and insert into a dictionary
        for word in range(len(words) + 2):
            print("Len ", len(words))
            print("word ", word)
            print(words[word])
            print("Range ", range(len(wordweights)+ 1))
            #print(wordweights[word])
            sentencepurpose[words[word]] = wordweights[word]
            print("Sentence ", sentencepurpose[words[word]])

        sentencepurpose = dict(sorted(sentencepurpose.items(), key=operator.itemgetter(1), reverse=True))
        #print("Purpose ", sentencepurpose)
        top_3_words = list(sentencepurpose)[:3]
        print("Purpose ", top_3_words)
        
    def __lemmatizerDoco(self, text):
        lemmatizer = self.nlp.Defaults.create_lemmatizer()
        lemm_ = [lemmatizer.lookup(word.lower()) for word in text]
        #print("LEMMA ", lemm_)       
        lemm_freq = Counter(lemm_)
        common_lemm = lemm_freq.most_common(10)
        print("Common Lemmaz ", common_lemm)

if __name__ == '__main__':
    print("Turbo")
    courtdoco = Document("usaassangejudgement.pdf")
