import pandas as pd
from math import pi
from collections import Counter
import spacy
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Doc, Span, Token
from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy import linalg
import numpy as np
import operator
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sb
import warnings

sb.set_theme(style="whitegrid")
np.set_printoptions(precision=1)
warnings.filterwarnings(action='once')

# Initialize spaCy
nlp = spacy.load("en_core_web_lg")
nlp.add_pipe('sentencizer')

# Constants
NUMBER_OF_TOPICS = 5

# Custom stop words and non-punct words
customize_stop_words = ['a.', 'b.', 'c.', 'i.', 'ii', 'iii', 'the', 'to', " \x0c", ' ', 
                       'Mr.', 'Dr.', 'v', 'of', 'case', 'section', 'defence', 'trial', 
                       'evidence', 'law', 'court', 'Court', 'criminal', 'Act', 'Article', 
                       'UK', 'extradition', 'offence', 'information', '“', '-v-', 'A.', 
                       'B.', '(', ')', 'wlr', 'wikileaks']
customize_non_punct = ['Ms.']

def initialize_nlp():
    """Initialize NLP with custom settings"""
    for w in customize_stop_words:
        nlp.vocab[w].is_stop = True
    for w in customize_non_punct:
        nlp.vocab[w].is_punct = False

def clean_text(text):
    """Clean and process text"""
    doc = nlp(text)
    clean_tokens = [t.text for t in doc if not t.is_stop and not t.whitespace_ and 
                   not t.text.isspace() and not t.is_punct and t.pos_ != "PRON"]
    return ' '.join([str(elem) for elem in clean_tokens if len(elem) > 2])

def analyze_dataframe(df, text_column='text'):
    """Main function to analyze text in DataFrame"""
    initialize_nlp()
    
    # Process each text in the DataFrame
    results = []
    for _, row in df.iterrows():
        text = row[text_column]
        clean_text_data = clean_text(text)
        clean_doc = nlp(clean_text_data)
        
        # Extract POS
        nouns = [t.lemma_ for t in clean_doc if t.pos_ == "NOUN"]
        verbs = [t.lemma_ for t in clean_doc if t.pos_ == "VERB"]
        adjectives = [t.lemma_ for t in clean_doc if t.pos_ == "ADJ"]
        
        # Tokenize and vectorize
        weights_dict = tokenize_document(clean_doc)
        
        # Topic modeling
        svd_topics = svd_decomposition(clean_doc)
        nmf_topics = nmf_decomposition(clean_doc)
        lda_topics = lda_decomposition(clean_doc)
        
        # Analyze topics
        common_topics = set(lda_topics) & set(nmf_topics) & set(svd_topics)
        main_topics = {k: weights_dict[k] for k in common_topics if k in weights_dict}
        
        # POS analysis
        verb_results = analyze_verbs(verbs)
        noun_results = analyze_nouns(nouns)
        adj_results = analyze_adjectives(adjectives)
        
        results.append({
            'text': text,
            'clean_text': clean_text_data,
            'nouns': noun_results,
            'verbs': verb_results,
            'adjectives': adj_results,
            'topics': main_topics,
            'weights': weights_dict
        })
    
    return results

def tokenize_document(doc):
    """Tokenize and vectorize document"""
    sents_list = [sent.text for sent in doc.sents]
    
    tfidf_vector = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, 
                                  norm=None, analyzer='word')
    model = tfidf_vector.fit(sents_list)
    transformed_model = model.transform(sents_list)
    
    return dict(zip(model.get_feature_names(), tfidf_vector.idf_))

def svd_decomposition(doc):
    """Perform SVD topic modeling"""
    sents_list = [sent.text for sent in doc.sents]
    bow_vector = CountVectorizer(min_df=0.001, max_df=0.95, stop_words='english')
    vectors = bow_vector.fit_transform(sents_list).todense()
    vocab = np.array(bow_vector.get_feature_names())
    
    U, s, Vh = linalg.svd(vectors, full_matrices=False)
    return get_topics(Vh[:NUMBER_OF_TOPICS], vocab)

def nmf_decomposition(doc):
    """Perform NMF topic modeling"""
    sents_list = [sent.text for sent in doc.sents]
    bow_vector = CountVectorizer(min_df=0.001, max_df=0.95, stop_words='english')
    vectors = bow_vector.fit_transform(sents_list).todense()
    vocab = np.array(bow_vector.get_feature_names())
    
    topic_model = decomposition.NMF(n_components=NUMBER_OF_TOPICS, random_state=1)
    topic_model.fit_transform(vectors)
    return get_topics(topic_model.components_, vocab)

def lda_decomposition(doc):
    """Perform LDA topic modeling"""
    sents_list = [sent.text for sent in doc.sents]
    bow_vector = CountVectorizer(min_df=0.001, max_df=0.95, stop_words='english')
    vectors = bow_vector.fit_transform(sents_list).todense()
    vocab = np.array(bow_vector.get_feature_names())
    
    topic_model = decomposition.LatentDirichletAllocation(
        n_components=NUMBER_OF_TOPICS, max_iter=10, 
        learning_method='online', verbose=True)
    topic_model.fit_transform(vectors)
    return get_topics(topic_model.components_, vocab)

def get_topics(vector, vocab, num_top_words=10):
    """Extract top words for topics"""
    top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-num_top_words-1:-1]]
    topic_words = ([top_words(t) for t in vector])
    return [' '.join(t) for t in topic_words]

def analyze_verbs(verbs):
    """Analyze verbs and create visualizations"""
    verb_freq = Counter(verbs)
    common_verbs = verb_freq.most_common(10)
    create_radar_chart(common_verbs, 'Top 10 Frequent Actions', 'Actions')
    create_bar_chart(common_verbs, 'Top 10 Frequent Actions', 'Actions')
    return common_verbs

def analyze_nouns(nouns):
    """Analyze nouns and create visualizations"""
    noun_freq = Counter(nouns)
    common_nouns = noun_freq.most_common(10)
    create_radar_chart(common_nouns, 'Top 10 Frequent Subjects', 'Subjects')
    create_bar_chart(common_nouns, 'Top 10 Frequent Subjects', 'Subjects')
    return common_nouns

def analyze_adjectives(adjectives):
    """Analyze adjectives and create visualizations"""
    adj_freq = Counter(adjectives)
    common_adjs = adj_freq.most_common(10)
    create_radar_chart(common_adjs, 'Top 10 Frequent Referrals', 'Referrals')
    create_bar_chart(common_adjs, 'Top 10 Frequent Referrals', 'Referrals')
    return common_adjs

def create_radar_chart(words, title, subject):
    """Create radar chart visualization"""
    fig, axes = plt.subplots(figsize=(9, 9))
    graph_data = {'group': ['A']}
    
    for word, freq in words:
        graph_data[word] = [freq]
    
    df = pd.DataFrame(graph_data)
    categories = list(df)[1:]
    N = len(categories)
    
    values = df.loc[0].drop('group').values.flatten().tolist()
    values += values[:1]
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    ax.set_rlabel_position(0)
    plt.yticks([20, 60, 100, 140, 180], ["20", "60", "100", "140", "180"], 
               color="grey", size=8)
    plt.ylim(0, max(values))
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    ax.fill(angles, values, 'b', alpha=0.1)
    ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                 horizontalalignment='center', verticalalignment='center')
    plt.savefig(f'{subject}_radar.png')
    plt.show()

def create_bar_chart(words, title, subject):
    """Create bar chart visualization"""
    fig, ax = plt.subplots(figsize=(11, 7))
    graph_data = {word: [freq] for word, freq in words}
    
    df = pd.DataFrame(graph_data)
    categories = list(df)[0:]
    values = df.loc[0].values.flatten().tolist()
    
    y_pos = np.arange(len(categories))
    plt.barh(categories, values)
    ax.invert_yaxis()
    ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                 horizontalalignment='center', verticalalignment='center')
    ax.set_ylabel(subject, fontweight='bold')
    ax.set_xlabel("Term Frequency", fontweight='bold')
    
    for i in ax.patches:
        plt.text(i.get_width()+0.2, i.get_y()+0.5, str(round(i.get_width(), 2)),
                fontsize=10, fontweight='bold', color='grey')
    
    fig.text(0.9, 0.15, 'Seyhan AI', fontsize=12, color='grey', 
             ha='right', va='bottom', alpha=0.7)
    plt.subplots_adjust(bottom=0.2, top=0.9)
    plt.savefig(f'{subject}.png')
    plt.show()

def plot_topics(topics_dict, title='Topics Analysis'):
    """Plot topics with their weights"""
    sorted_topics = dict(sorted(topics_dict.items(), key=lambda item: item[1]))
    x, y = zip(*sorted_topics.items())
    
    df = pd.DataFrame({"Topics": x, "Inverse Term Frequency Ranks": y})
    graph = sb.PairGrid(df, x_vars=["Inverse Term Frequency Ranks"], y_vars=["Topics"],
                       height=10, aspect=0.8)
    graph.map(sb.stripplot, size=12, orient="h", jitter=False,
             palette="flare_r", linewidth=1, edgecolor="w")
    
    plt.title(title, weight='bold', fontdict={'size': 11})
    plt.subplots_adjust(left=0.16, bottom=0.16, top=0.9)
    plt.show()

# Example usage:
if __name__ == '__main__':
    # Create sample DataFrame
    data = {
        'text': [
            "This document discusses Python programming and data analysis techniques.",
            "The court case involved criminal charges under the UK Act.",
            "Machine learning models were used for evidence analysis in the trial."
        ],
        'page_number': [1, 2, 3]
    }
    df = pd.DataFrame(data)
    
    # Analyze the DataFrame
    analysis_results = analyze_dataframe(df)
    
    # For the first document's results
    first_result = analysis_results[0]
    print("Top Nouns:", first_result['nouns'])
    print("Top Verbs:", first_result['verbs'])
    print("Main Topics:", first_result['topics'])
    
    # Plot topics for the first document
    plot_topics(first_result['topics'])