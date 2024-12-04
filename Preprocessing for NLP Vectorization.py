#apply this function to any text or any field in a dataframe to set it up for ntlk factorization.
# make sure the valid libraries and dependencies are installed

from bs4 import BeautifulSoup
import unicodedata
import re
import nltk
from nltk.corpus import wordnet
import spacy
import string
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

nlp = spacy.load('en_core_web_sm')
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')

def preprocess_text(text):
    def simple_stemmer(text):
        ps = nltk.porter.PorterStemmer()
        return ' '.join([ps.stem(word) for word in text.split()])

    def remove_hastags_and_link(text):
        return ' '.join(re.sub("(#[A-Za-z0-9]+)|(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|[^a-zA-Z0-9\s]", " ", text).split())

    def remove_digits(text):
        return ''.join([c for c in text if c not in string.digits])

    def strip_html_tags(text):
        soup = BeautifulSoup(text, "html.parser")
        [s.extract() for s in soup(['iframe', 'script'])]
        stripped_text = soup.get_text()
        return re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)

    def remove_accented_chars(text):
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    def remove_special_characters(text, remove_digits=False):
        pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
        return re.sub(pattern, "", text)

    def remove_extra_lines(text):
        return re.sub(r'[\r|\n|\r\n]+', ' ', text)

    def change_to_lower(text):
        return text.lower()

    def remove_extra_space(text):
        return re.sub(' +', ' ', text)

    def remove_all_space(text):
        return re.sub(' ', '', text)

    def remove_amp(text):
        return re.sub(r'amp', '', text)

    def lemmatize_text(text):
        text = nlp(text)
        return ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])

    def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list):
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        if is_lower_case:
            filtered_tokens = [token for token in tokens if token not in stopwords]
        else:
            filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
        return ' '.join(filtered_tokens)

    def tokenize_text(text):
        sentences = nltk.sent_tokenize(text)
        word_tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
        return word_tokens
   
    def remove_repeated_characters(tokens):
        repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
        match_substitution = r'\1\2\3'

    def remove_repeated_characters(tokens):
        repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
        match_substitution = r'\1\2\3'

        def replace(old_word):
            if wordnet.synsets(old_word):
                return old_word
            new_word = repeat_pattern.sub(match_substitution, old_word)
            return replace(new_word) if new_word != old_word else new_word

        # Flatten the list of lists into a single list
        all_tokens = [token for sublist in tokens for token in sublist]
        correct_tokens = [replace(word) for word in all_tokens]
        return ''.join(correct_tokens)

    # Applying all functions
    text = simple_stemmer(text)
    text = remove_hastags_and_link(text)
    text = remove_digits(text)
    text = strip_html_tags(text)
    text = remove_accented_chars(text)
    text = remove_special_characters(text)
    text = remove_extra_lines(text)
    text = change_to_lower(text)
    text = remove_extra_space(text)
    text = remove_all_space(text)
    text = remove_amp(text)
    text = lemmatize_text(text)
    text = remove_stopwords(text)
    tokens = tokenize_text(text)
    text = remove_repeated_characters(tokens)

    return text
