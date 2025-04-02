from collections import Counter
import pandas as pd
import numpy as np
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy import linalg
import matplotlib.pyplot as plt
import seaborn as sb
import warnings
from io import StringIO

# Set up for visualization and suppress warnings
sb.set_theme(style="whitegrid")
np.set_printoptions(precision=1)
warnings.filterwarnings(action='once')

# Define global variables for NLP and analysis tools
nlp = spacy.load("en_core_web_lg")
tokenizer = Tokenizer(nlp.vocab)
nlp.add_pipe(nlp.create_pipe('sentencizer'))

number_of_topics = 5
svd_topics = []
nmf_topics = []
lda_topics = []

common_verbs = []
common_nouns = []
common_adjs = []

topics = {}  
weights_dict = {}

# Helper function for text analysis on DataFrame
def text_analysis_from_dataframe(df, column_name):
    text = ' '.join(df[column_name].dropna())  # Combine all text from the specified column
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

    tokenize_doc(clean_doc)
    svd_decomp(clean_doc)
    nmf_decomp(clean_doc)
    lda_decomp(clean_doc)
    topic_analysis()
    plot_topics()

# Tokenize the document
def tokenize_doc(doc):
    sents_list = [sent.text for sent in doc.sents]
    
    tfidf_vector = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word')
    model = tfidf_vector.fit(sents_list)
    transformed_model = model.transform(sents_list)
    
    global weights_dict
    weights_dict = dict(zip(model.get_feature_names(), tfidf_vector.idf_))
    print("Weight Dict ", weights_dict)

# SVD decomposition
def svd_decomp(doc):
    sents_list = [sent.text for sent in doc.sents]
    bow_vector = CountVectorizer(min_df=0.001, max_df=0.95, stop_words='english')
    vectors = bow_vector.fit_transform(sents_list).todense()
    vocab = np.array(bow_vector.get_feature_names())
    
    U, s, Vh = linalg.svd(vectors, full_matrices=False)
    
    topics = get_topics(Vh[:number_of_topics], vocab)
    tokenize_topics(topics, "SVD")

# NMF decomposition
def nmf_decomp(doc):
    sents_list = [sent.text for sent in doc.sents]
    bow_vector = CountVectorizer(min_df=0.001, max_df=0.95, stop_words='english')
    vectors = bow_vector.fit_transform(sents_list).todense()
    vocab = np.array(bow_vector.get_feature_names())
    
    topic_model = decomposition.NMF(n_components=number_of_topics, random_state=1)
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
    topic_model = decomposition.LatentDirichletAllocation(n_components=number_of_topics, max_iter=10, learning_method='online', verbose=True)
    lda_fit = topic_model.fit_transform(vectors)
    topic_model_comps = topic_model.components_
    
    topics = get_topics(topic_model_comps, vocab)
    tokenize_topics(topics, "LDA")

# Helper function to tokenize topics
def tokenize_topics(topics, model_type):
    list_to_str = ' '.join([str(elem) for elem in topics if len(elem) > 2])
    doc = nlp(list_to_str)
    
    global lda_topics, nmf_topics, svd_topics
    for sent in doc:
        if model_type == "LDA":
            lda_topics.append(sent.text)
        elif model_type == "NMF":
            nmf_topics.append(sent.text)
        elif model_type == "SVD":
            svd_topics.append(sent.text)

# Topic analysis
def topic_analysis():
    global topics
    topics = set(lda_topics) & set(nmf_topics) & set(svd_topics)

# Helper function to get topics
def get_topics(vector, vocab):
    num_top_words = 10
    top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-num_top_words-1:-1]]
    topic_words = ([top_words(t) for t in vector])
    return [' '.join(t) for t in topic_words]

# Plot the topics
def plot_topics():
    # Example plot, can be adjusted according to the desired visualization
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(common_verbs)), [item[1] for item in common_verbs])
    plt.xticks(range(len(common_verbs)), [item[0] for item in common_verbs], rotation=90)
    plt.title("Top 10 Frequent Verbs")
    plt.show()

# Example usage:
# Assuming 'df' is your Pandas DataFrame and 'text_column' is the column containing text
# Example DataFrame:
data = {
    'text_column': [
        "The law of torts governs civil wrongs, including negligence and defamation.",
        "In the case of a breach of contract, the aggrieved party may seek remedies.",
        "The criminal law aims to punish offenses and deter criminal behavior."
    ]
}
df = pd.DataFrame(data)

# Call the function to analyze text
text_analysis_from_dataframe(df, 'text_column')
