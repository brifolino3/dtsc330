import nltk 
import fasttext
import spacy

# NLTK is a set of rules that are useful for parsing language
# think a list of every type of punctuation as an example

# spacy is built on neural networks
# this is used for  things like parsing out noun phrases

# break apart text into sentences
def break_sentences(txt: str) -> list[str]:
    """break text into a list of sentences"""
    return nltk.tokenize.sent_tokenize(txt)


print(break_sentences("This is a test."))

# find noun phrases
class NounParser():
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def phrases(self, txt: str -> list[str]):
        """convert text into noun phrases"""
        doc = self. nlp(txt.lower()) # keep everything in one case for later matching
        words = []
        for phrase in doc.noun_chunks:
            word = phrase.text.strip()
            if not (len(word) == 0 or word.isspace()):
                words.append(word)
        return words
    
np = NounParser()
print(np.phrases("HELLOOOOOOO"))

ft_model = fasttext.load_model('data/cc.en.50.bin')
print(ft_model.get_sentence_vector("BLAH"))