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

from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy import linalg
import numpy as np
import operator
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sb
import pandas as pd
import io
import os
sb.set_theme(style="whitegrid")
np.set_printoptions(precision=1)
import warnings; warnings.filterwarnings(action='once')

# Define global variables for NLP and analysis tools
nlp = spacy.load("en_core_web_lg")
tokenizer = Tokenizer(nlp.vocab)
nlp.add_pipe(nlp.create_pipe('sentencizer'))

numberofTopics = 5
svdTopics = []
nmfTopics = []
ldaTopics = []

common_verbs = []
common_nouns = []
common_adjs = []

Topics = {}  
weightsDict = {}

# Helper function to convert PDF to text
def convert_to_text(file_name):
    file_handle = io.StringIO()
    resource_manager = PDFResourceManager()
    converter = TextConverter(resource_manager, file_handle)
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    
    with open(file_name, 'rb') as fh:
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            page_interpreter.process_page(page)
        
        text = file_handle.getvalue()  # whole document in text
    converter.close()
    file_handle.close()
    
    text_analysis(text)

# Helper function for text analysis
def text_analysis(text):
    # Add law jargon and terms to stop words
    customize_stop_words = ['a.', 'b.', 'c.', 'i.', 'ii', 'iii', 'the', 'to', " \x0c", ' ', 'Mr.', 'Dr.', 'v', 'of', 'case', 'section', 'defence',
    'trial', 'evidence', 'law', 'court', 'Court', 'criminal', 'Act', 'Article', 'UK', 'extradition', 'offence', 'information',
    '“', '-v-', 'A.', 'B.', '(', ')', 'wlr', 'wikileaks']
    for w in customize_stop_words:
        nlp.vocab[w].is_stop = True
    
    customize_non_punct = ['Ms.']
    for w in customize_non_punct:
        nlp.vocab[w].is_punct = False
    
    doc = nlp(text)
    
    clean_doc = [t.text for t in doc if t.is_stop != True and t.whitespace_ != True and t.text.isspace() != True and t.is_punct != True and t.pos_ != "-PRON-"]
    list_to_str = ' '.join([str(elem) for elem in clean_doc if len(elem) > 2])
    clean_doc = nlp(list_to_str)
    
    tokenize_doco(clean_doc)
    svd_decomp(clean_doc)
    nmf_decomp(clean_doc)
    lda_decomp(clean_doc)
    topic_analysis()
    plot_topics()

# Tokenize the document
def tokenize_doco(doc):
    sents_list = [sent.text for sent in doc.sents]
    
    tfidf_vector = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word')
    model = tfidf_vector.fit(sents_list)
    transformed_model = model.transform(sents_list)
    
    global weightsDict
    weightsDict = dict(zip(model.get_feature_names(), tfidf_vector.idf_))
    print("Weight Dict ", weightsDict)

# SVD decomposition
def svd_decomp(doc):
    sents_list = [sent.text for sent in doc.sents]
    bow_vector = CountVectorizer(min_df=0.001, max_df=0.95, stop_words='english')
    vectors = bow_vector.fit_transform(sents_list).todense()
    vocab = np.array(bow_vector.get_feature_names())
    
    U, s, Vh = linalg.svd(vectors, full_matrices=False)
    
    topics = get_topics(Vh[:numberofTopics], vocab)
    tokenize_topics(topics, "SVD")

# NMF decomposition
def nmf_decomp(doc):
    sents_list = [sent.text for sent in doc.sents]
    bow_vector = CountVectorizer(min_df=0.001, max_df=0.95, stop_words='english')
    vectors = bow_vector.fit_transform(sents_list).todense()
    vocab = np.array(bow_vector.get_feature_names())
    
    topic_model = decomposition.NMF(n_components=numberofTopics, random_state=1)
    fitted_model = topic_model.fit_transform(vectors)
    topic_model_comps = topic_model.components_
    
    topics = get_topics(topic_model_comps, vocab)
    tokenize_topics(topics, "NMF")

# LDA decomposition
def lda_decomp(doc):
    sents_list = [sent.text for sent in doc.sents]
    bow_vector = CountVectorizer(min_df=0.001, max_df=0.95, stop_words='english')
    vectors = bow_vector.fit_transform(sents_list).todense()
    
    vocab = np.array(bow_vector.get_feature_names())
    topic_model = decomposition.LatentDirichletAllocation(n_components=numberofTopics, max_iter=10, learning_method='online', verbose=True)
    lda_fit = topic_model.fit_transform(vectors)
    topic_model_comps = topic_model.components_
    
    topics = get_topics(topic_model_comps, vocab)
    tokenize_topics(topics, "LDA")

# Helper function to tokenize topics
def tokenize_topics(topics, model_type):
    list_to_str = ' '.join([str(elem) for elem in topics if len(elem) > 2])
    doc = nlp(list_to_str)
    
    global ldaTopics, nmfTopics, svdTopics
    for sent in doc:
        if model_type == "LDA":
            ldaTopics.append(sent.text)
        elif model_type == "NMF":
            nmfTopics.append(sent.text)
        elif model_type == "SVD":
            svdTopics.append(sent.text)

# Topic analysis
def topic_analysis():
    global Topics
    Topics = set(ldaTopics) & set(nmfTopics) & set(svdTopics)

# Helper function to get topics
def get_topics(vector, vocab):
    num_top_words = 10
    top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-num_top_words-1:-1]]
    topic_words = ([top_words(t) for t in vector])
    return [' '.join(t) for t in topic_words]

# Analyze verbs
def verb_analysis(verbs):
    verb_freq = Counter(verbs)
    global common_verbs
    common_verbs = verb_freq.most_common(10)

# Analyze nouns
def noun_analysis(nouns):
    noun_freq = Counter(nouns)
    global common_nouns
    common_nouns = noun_freq.most_common(10)

# Analyze adjectives
def adjective_analysis(adjectives):
    adj_freq = Counter(adjectives)
    global common_adjs
    common_adjs = adj_freq.most_common(10)

# Plot the topics
def plot_topics():
    # Example plot, can be adjusted according to the desired visualization
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(common_verbs)), [item[1] for item in common_verbs])
    plt.xticks(range(len(common_verbs)), [item[0] for item in common_verbs], rotation=90)
    plt.title("Top 10 Frequent Verbs")
    plt.show()

# Example call to process a PDF file
convert_to_text('sample.pdf')
