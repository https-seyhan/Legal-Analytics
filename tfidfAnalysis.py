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
from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction import stop_words
from scipy import linalg
import numpy as np
import operator
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import io
import os
sb.set_theme(style="whitegrid")
np.set_printoptions(precision=2)
    
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

    numberofTopics = 5
    svdTopics = []
    nmfTopics = []
    ldaTopics = []

    common_verbs = []
    common_nouns = []
    common_adjs = []

    Topics = {}  
    weightsDict = {}  

    #fig, (ax1, ax2, ax3) = plt.subplot(nrows=1, ncols=3, polar=True)
    
    
    def __init__(self, fileName):
        
        self.__convertToText(fileName)
        
    def __convertToText(self, fileName):
        list = []
        #print ("file Name ", fileName)
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
        #print(type(text))
        
        # Add law jargon and terms to stop words
        customize_stop_words = ['a.', 'b.', 'c.', 'i.', 'ii', 'iii', 
        'the', 'to', " \x0c", ' ', 'Mr.', 'Dr.', 'v', 'of', 'case', 'section', 'defence',
        'trial', 'evidence', 'law', 'court', 'Court', 'criminal', 'Act', 'Article', 'UK','extradition', 'offence', 'information',
        '“', '-v-', 'A.', 'B.', '(', ')', 'wlr', 'wikileaks'
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
        
        # convert List to String not include strings less then 3
        listToStr = ' '.join([str(elem) for elem in cleanDoc if len(elem) > 2]) 
        #print(listToStr) # Print clean data
        cleanDoc = self.nlp(listToStr)
        #print(" Clean Doc ", cleanDoc)
        self.__tokenizeDoco(cleanDoc)
        self.__svdDecomp(cleanDoc)
        self.__NMFDecomp(cleanDoc)
        self.__LDADecomp(cleanDoc)
        self.__topicAnalysis()
        self.__plotTopics()
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
        self.__verbAnalysis(verbs)
        self.__nounAnalysis(nouns)
        #self.__lemmatizerDoco(nouns)
        #self.__lemmatizerDoco(verbs)
        #self.__lemmatizerDoco(adjectives)
        #self.__lemmatizerDoco(others)
        self.__adjectiveAnalysis(adjectives)
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
        #tokenize = self.tokenizer(text)
        #sentences = [sent.string.strip() for sent in text.sents]
        #print(sentences)
        sents_list = []
        for sent in doc.sents:
            sents_list.append(sent.text)
        #print(sents_list)

        #tfidf_vector = TfidfVectorizer()
        tfidf_vector = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word')
        #print(tfidf_vector)
        #model = tfidf_vector.fit_transform(sents_list)
        model = tfidf_vector.fit(sents_list)
        transformed_model = model.transform(sents_list) #Transform documents to document-term matrix.
        
        
        #weight_dict = dict(zip(tfidf_vector.get_feature_names(), tfidf_vector.idf_))
        self.weightsDict = dict(zip(model.get_feature_names(), tfidf_vector.idf_))
        print("Weight Dict ", self.weightsDict)
        #print(type(sents_list))
        
        #Weight of words per document
        #print( "Word weight per document ", transformed_model.toarray())
        max_val = transformed_model.max(axis=0).toarray().ravel()
        
        sort_by_tfidf = max_val.argsort()
        
        feature_names = np.array(tfidf_vector.get_feature_names())
        #print("Features with lowest tfidf:\n{}".format(feature_names[sort_by_tfidf[:10]]))

        print("\nFeatures with highest tfidf: \n{}".format(feature_names[sort_by_tfidf[-10:]]))

    
    def __svdDecomp(self, doc):
    
        sents_list = []
        bow_vector = CountVectorizer(min_df =0.001, max_df=0.95, stop_words='english') # Convert a collection of text documents to a matrix of token counts        
        for sent in doc.sents:
            sents_list.append(sent.text)
        #print(bow_vector) 
        vectors = bow_vector.fit_transform(sents_list).todense()
        #print (" Vectors ", vectors)
        vocab = np.array(bow_vector.get_feature_names())
        U, s, Vh = linalg.svd(vectors, full_matrices=False)
        #plt.plot(s);
        #plt.plot(s[:10])
        #plt.show()
        #plt.plot(Vh[:10])
        #plt.show()
        topics = self.__get_topics(Vh[:self.numberofTopics], vocab)
        #print("Topics ", topics)
        self.__tokenizeTopics(topics, "SVD")
        

    def __NMFDecomp(self, doc):
 
        sents_list = []
        bow_vector = CountVectorizer(min_df =0.001, max_df=0.95, stop_words='english') # Convert a collection of text documents to a matrix of token counts        
        for sent in doc.sents:
            sents_list.append(sent.text)
        #print(bow_vector) 
        vectors = bow_vector.fit_transform(sents_list).todense()
        #print (" Vectors ", vectors)
        vocab = np.array(bow_vector.get_feature_names())
        m,n=vectors.shape
        
        topicModel = decomposition.NMF(n_components= self.numberofTopics, random_state=1)
        fittedModel = topicModel.fit_transform(vectors)
        topicModelComps = topicModel.components_
        topics = self.__get_topics(topicModelComps, vocab)
        #print("Topics ", topics)
        self.__tokenizeTopics(topics, "NMF")

    def __LDADecomp(self, doc):

        sents_list = []
        bow_vector = CountVectorizer(min_df =0.001, max_df=0.95, stop_words='english') # Convert a collection of text documents to a matrix of token counts        
        for sent in doc.sents:
            sents_list.append(sent.text)
        #print(bow_vector) 
        vectors = bow_vector.fit_transform(sents_list).todense()
        
        vocab = np.array(bow_vector.get_feature_names())
        m,n=vectors.shape

        topicModel = decomposition.LatentDirichletAllocation(n_components=self.numberofTopics, max_iter=10, learning_method='online',verbose=True)
        #data_lda = lda.fit_transform(data_vectorized)
        lda_fit = topicModel.fit_transform(vectors) #Learn the vocabulary dictionary and return document-term matrix
        topicModelComps = topicModel.components_
        #print(topicModelComps)
        
        topics = self.__get_topics(topicModelComps, vocab)

        self.__tokenizeTopics(topics, "LDA")

    def __tokenizeTopics(self, topics, modeltype):

        # convert List to String not include strings less then 3
        listToStr = ' '.join([str(elem) for elem in topics if len(elem) > 2]) 
        #print(listToStr) # Print clean data
        
        doc = self.nlp(listToStr)
 
        for sent in doc:
            if modeltype == "LDA":
                self.ldaTopics.append(sent.text)
            elif modeltype == "NMF":
                self.nmfTopics.append(sent.text)
            elif modeltype == "SVD":
                self.svdTopics.append(sent.text)        

    def __topicAnalysis(self):
        self.Topics = set(self.ldaTopics) & set(self.nmfTopics) & set(self.svdTopics)
        
    def __topTopics(self, model, vectorizer, top_n = 5):
        for idx, topic in enumerate(model.components_):
            print("Topic %d:" % (idx))
            print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]]) 

    def __get_topics(self, vector, vocab):
        num_top_words=10
        top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-num_top_words-1:-1]]
        topic_words = ([top_words(t) for t in vector])
        return [' '.join(t) for t in topic_words] 
     

    def __getPurpose(self, model, clean_text, sentenceNum):
                
        # get weights of words
        wordweights = model[sentenceNum].data
        
        #print(" Clean Text ", clean_text)
        words = clean_text.split(" ")
        #remove duplicates in list
        words = list(dict.fromkeys(words))

        sentencepurpose = {}
        word = 0
        print("Sentence Size ", len(words), '\n')
        print("Weight Size ", len(wordweights), '\n')
        #get tfidf vectors and insert into a dictionary
        while len(words) > word:
            sentencepurpose[words[word]] = wordweights[word]          
            word += 1
        print("END!!!!!!!!")

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

        # Get Bag of Words (BoW) of top 10 words
    def __verbAnalysis(self, verbs):
        verb_freq = Counter(verbs)
        self.common_verbs = verb_freq.most_common(10)
        self.__radar(self.common_verbs, 'Top 10 Frequent Verbs')
        self.__bar(self.common_verbs, 'Top 10 Frequent Verbs', 'Verbs')

    def __nounAnalysis(self, nouns):
        noun_freq = Counter(nouns)
        self.common_nouns = noun_freq.most_common(10)       
        self.__radar(self.common_nouns, 'Top 10 Frequent Subjects')
        self.__bar(self.common_nouns, 'Top 10 Frequent Subjects', 'Nouns')
        
    def __adjectiveAnalysis(self, adjectives):
        adj_freq = Counter(adjectives)
        self.common_adjs = adj_freq.most_common(10)
        
        self.__radar(self.common_adjs, 'Top 10 Frequent Referrals')
        self.__bar(self.common_adjs, 'Top 10 Frequent Referrals', 'Adjectives' )
    
    def __otherAnalysis(self, others):
        oth_freq = Counter(others)
        common_oths = oth_freq.most_common(10)
        #print("Common other ", common_oths)
        
    def __wordAnalysis(self, tokens, nouns, verbs, adjectives, docents):
        
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
    
    def __plotTopics(self):

        mainTopics = {}

        for key in self.Topics:
            #for key1 in self.weightsDict:

            if key in self.weightsDict:
                #print(key, self.weightsDict[key])
                mainTopics[key] = self.weightsDict[key]

        tt = dict(sorted(mainTopics.items(), key=lambda item: item[1])) # sort topics with their idf

        x, y = zip(*tt.items()) # unpack a list of pairs into two tuples

        df = pd.DataFrame({"topic":x, 
                          "rank":y})

        g = sb.PairGrid(df, x_vars= ["rank"] , y_vars=["topic"],
                        height=10, aspect= 0.8)

        g.map(sb.stripplot, size=12, orient="h", jitter=False,
              palette="flare_r", linewidth=1, edgecolor="w")

        plt.subplots_adjust(left = 0.16, bottom=0.16, top=0.99)
        plt.show()

    def __radar(self, words, title):

        fig, axes = plt.subplots(figsize=(9, 9))
        fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
        
        graphdata = {}
        graphdata['group'] = ['A']
                
        for _ in range(len(words)):
            #print(words[_][0])
            graphdata[words[_][0]]= [words[_][1]]
                
        dataframe = pd.DataFrame(graphdata)
           
        categories=list(dataframe)[1:]
        N = len(categories)
        
        values=dataframe.loc[0].drop('group').values.flatten().tolist()
        values += values[:1]
        
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        ax = plt.subplot(111, polar=True)
        plt.xticks(angles[:-1], categories, color='grey', size=10)
        ax.set_rlabel_position(0)
        plt.yticks([20, 60,  100, 140, 180], 
        ["20", "60", "100", "140", "180"], color="grey", size=8)
        plt.ylim(0,max(values))
        ax.plot(angles, values, linewidth=1, linestyle='solid')
        ax.fill(angles, values, 'b', alpha=0.1)
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        plt.show()

    def __bar(self, words, title, subject):

        figs = plt.subplots(figsize=(9, 9))
 
        graphdata = {}
        
        for _ in range(len(words)):
            #print(words[_][0])
            graphdata[words[_][0]]= [words[_][1]]
        
        dataframe = pd.DataFrame(graphdata)
        categories=list(dataframe)[0:]
        values=dataframe.loc[0].values.flatten().tolist()

        
        y_pos = np.arange(len(categories))
        ax = plt.subplot(111)

        plt.bar(categories,values, align='center', alpha=0.5, color=(0.1, 0.1, 0.1, 0.1))
        plt.xticks(y_pos, categories,  rotation=90)        
        
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        ax.set_ylabel("Term Frequency", fontweight ='bold')
        ax.set_xlabel(subject, fontweight ='bold')
        plt.subplots_adjust(bottom=0.2, top=0.9)
        plt.show()

if __name__ == '__main__':
    
    courtdoco = Document("usaassangejudgement.pdf")
