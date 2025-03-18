import re
from typing import List, Tuple, Set, Dict
import unicodedata
from collections import defaultdict
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

class SpellChecker:
    def __init__(self, dictionary_file: str = "data/medium_corpus.txt", real_word_threshold: float = 0.00001):
        """
        Initialize the spell checker with a dictionary file.
        
        Args:
            dictionary_file: Path to the dictionary file
            real_word_threshold: Threshold for real-word error detection (default: 0.00001)
        """
        self.dictionary = self._load_dictionary(dictionary_file)
        self.word_frequencies = defaultdict(int)  # Track individual word frequencies
        self.bigrams = defaultdict(int)
        self.right_bigrams = defaultdict(int)
        self.trigrams = defaultdict(int)
        self.right_trigrams = defaultdict(int)  # New: right trigrams
        self.real_word_threshold = real_word_threshold
        self._load_ngrams(dictionary_file)
        
        # Initialize confusion sets
        self.confusion_sets = {
            'their': {'there', 'they\'re'},
            'there': {'their', 'they\'re'},
            'they\'re': {'their', 'there'},
            'your': {'you\'re'},
            'you\'re': {'your'},
            'its': {'it\'s'},
            'it\'s': {'its'},
            'to': {'too', 'two'},
            'too': {'to', 'two'},
            'two': {'to', 'too'},
            'than': {'then'},
            'then': {'than'},
            'affect': {'effect'},
            'effect': {'affect'},
            'accept': {'except'},
            'except': {'accept'},
            'weather': {'whether'},
            'whether': {'weather'},
            'principal': {'principle'},
            'principle': {'principal'},
            'stationary': {'stationery'},
            'stationery': {'stationary'},
            'complement': {'compliment'},
            'compliment': {'complement'},
            'desert': {'dessert'},
            'dessert': {'desert'},
            'loose': {'lose'},
            'lose': {'loose'},
            'passed': {'past'},
            'past': {'passed'},
            'peace': {'piece'},
            'piece': {'peace'},
            'plain': {'plane'},
            'plane': {'plain'},
            'right': {'write'},
            'write': {'right'},
            'sight': {'site', 'cite'},
            'site': {'sight', 'cite'},
            'cite': {'sight', 'site'},
            'threw': {'through'},
            'through': {'threw'},
            'waist': {'waste'},
            'waste': {'waist'},
            'weak': {'week'},
            'week': {'weak'},
            'wear': {'where'},
            'where': {'wear'},
            'which': {'witch'},
            'witch': {'which'},
            'whose': {'who\'s'},
            'who\'s': {'whose'}
        }
        
        # Initialize POS-specific confusion sets
        self.pos_confusion_sets = {
            'NN': {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'},  # Noun vs Verb
            'VB': {'NN', 'NNS', 'NNP', 'NNPS'},  # Verb vs Noun
            'JJ': {'RB'},  # Adjective vs Adverb
            'RB': {'JJ'},  # Adverb vs Adjective
            'IN': {'CC'},  # Preposition vs Conjunction
            'CC': {'IN'},  # Conjunction vs Preposition
            'DT': {'WDT'},  # Determiner vs Wh-determiner
            'WDT': {'DT'},  # Wh-determiner vs Determiner
            'PRP': {'PRP$'},  # Personal pronoun vs Possessive pronoun
            'PRP$': {'PRP'}  # Possessive pronoun vs Personal pronoun
        }
        
    def _preprocess_word(self, word: str) -> str:
        """
        Preprocess a word by:
        1. Converting to lowercase
        2. Removing special characters
        3. Normalizing unicode characters
        4. Removing extra whitespace
        5. Removing numbers
        """
        # Convert to lowercase
        word = word.lower()
        
        # Normalize unicode characters (e.g., convert é to e)
        word = unicodedata.normalize('NFKD', word).encode('ASCII', 'ignore').decode('ASCII')
        
        # Remove numbers and special characters
        word = re.sub(r'[0-9]', '', word)  # Remove numbers
        word = re.sub(r'[^\w\s]', '', word)  # Remove special characters
        word = word.strip()
        
        return word
    
    def _load_dictionary(self, dictionary_file: str) -> Set[str]:
        """
        Load and preprocess words from dictionary file into a set for O(1) lookup.
        """
        processed_words = set()
        
        with open(dictionary_file, "r", encoding='utf-8') as f:
            for line in f:
                # Split line into words and process each word
                words = line.strip().split()
                for word in words:
                    processed_word = self._preprocess_word(word)
                    if processed_word:  # Only add non-empty words
                        processed_words.add(processed_word)
        
        return processed_words
    
    def _load_ngrams(self, dictionary_file: str) -> None:
        """Load and count bigrams, right bigrams, and trigrams from the dictionary file."""
        total_words = 0
        with open(dictionary_file, "r", encoding='utf-8') as f:
            for line in f:
                words = [self._preprocess_word(word) for word in line.strip().split()]
                words = [w for w in words if w]  # Remove empty words
                
                # Count individual word frequencies
                for word in words:
                    self.word_frequencies[word] += 1
                    total_words += 1
                
                # Count left bigrams
                for i in range(len(words) - 1):
                    self.bigrams[(words[i], words[i + 1])] += 1
                
                # Count right bigrams
                for i in range(1, len(words)):
                    self.right_bigrams[(words[i-1], words[i])] += 1
                
                # Count left trigrams
                for i in range(len(words) - 2):
                    self.trigrams[(words[i], words[i + 1], words[i + 2])] += 1
                
                # Count right trigrams
                for i in range(2, len(words)):
                    self.right_trigrams[(words[i-2], words[i-1], words[i])] += 1
        
        # Store total word count for probability calculations
        self.total_words = total_words
    
    def _get_word_probability(self, word: str, context: List[str], next_words: List[str] = None) -> float:
        """
        Calculate the probability of a word occurring in the given context using n-grams.
        Now considers both left and right context and word frequency.
        """
        if not context and not next_words:
            # If no context, return unigram probability
            return self.word_frequencies.get(word, 0) / self.total_words
            
        word = self._preprocess_word(word)
        context = [self._preprocess_word(w) for w in context]
        context = [w for w in context if w]  # Remove empty words
        
        if next_words:
            next_words = [self._preprocess_word(w) for w in next_words]
            next_words = [w for w in next_words if w]
        
        # Calculate unigram probability
        unigram_prob = self.word_frequencies.get(word, 0) / self.total_words
        
        # Try left trigram first if we have enough context
        if len(context) >= 2:
            trigram = tuple(context[-2:] + [word])
            if trigram in self.trigrams:
                trigram_prob = self.trigrams[trigram] / sum(self.trigrams.values())
                
                # If we have right context, combine with right trigram probability
                if next_words and len(next_words) >= 2:
                    right_trigram = tuple([word] + next_words[:2])
                    if right_trigram in self.right_trigrams:
                        right_trigram_prob = self.right_trigrams[right_trigram] / sum(self.right_trigrams.values())
                        return 0.5 * trigram_prob + 0.3 * right_trigram_prob + 0.2 * unigram_prob
                return 0.8 * trigram_prob + 0.2 * unigram_prob
        
        # Try right trigram if we have enough right context
        if next_words and len(next_words) >= 2:
            right_trigram = tuple([word] + next_words[:2])
            if right_trigram in self.right_trigrams:
                right_trigram_prob = self.right_trigrams[right_trigram] / sum(self.right_trigrams.values())
                return 0.8 * right_trigram_prob + 0.2 * unigram_prob
        
        # Try left bigram
        if len(context) >= 1:
            left_bigram = tuple(context[-1:] + [word])
            if left_bigram in self.bigrams:
                left_prob = self.bigrams[left_bigram] / sum(self.bigrams.values())
                
                # If we have right context, combine with right bigram probability
                if next_words and len(next_words) > 0:
                    right_bigram = tuple([word] + next_words[:1])
                    if right_bigram in self.right_bigrams:
                        right_prob = self.right_bigrams[right_bigram] / sum(self.right_bigrams.values())
                        return 0.5 * left_prob + 0.3 * right_prob + 0.2 * unigram_prob
                return 0.8 * left_prob + 0.2 * unigram_prob
        
        # Try right bigram if we have right context
        if next_words and len(next_words) > 0:
            right_bigram = tuple([word] + next_words[:1])
            if right_bigram in self.right_bigrams:
                right_prob = self.right_bigrams[right_bigram] / sum(self.right_bigrams.values())
                return 0.8 * right_prob + 0.2 * unigram_prob
        
        return unigram_prob
    
    def _get_word_pos(self, word: str, context: List[str] = None) -> str:
        """
        Get the POS tag for a word using NLTK's pos_tag.
        If context is provided, uses it to improve POS tagging accuracy.
        """
        if context:
            # Create a sentence with context and target word
            sentence = ' '.join(context + [word])
        else:
            sentence = word
            
        # Tokenize and tag
        tokens = word_tokenize(sentence)
        tags = pos_tag(tokens)
        
        # Return the tag for the target word
        return tags[-1][1] if context else tags[0][1]
    
    def check_text(self, text: str) -> List[Tuple[str, int, int, str]]:
        """
        Check text for spelling errors and real-word errors.
        Returns a list of tuples containing (word, start_position, end_position, error_type).
        error_type can be 'non_word', 'real_word', or 'confusion_set'.
        """
        # Split text into words while preserving positions
        words = []
        for match in re.finditer(r'\b\w+\b', text):
            word = match.group()
            start, end = match.span()
            words.append((word, start, end))
        
        errors = []
        for i, (word, start, end) in enumerate(words):
            processed_word = self._preprocess_word(word)
            
            # Check for non-word errors
            if processed_word not in self.dictionary:
                errors.append((word, start, end, 'non_word'))
                continue
            
            # Get context for POS tagging and probability calculation
            context = [w for w, _, _ in words[max(0, i-2):i]]  # Get previous 2 words
            next_words = [w for w, _, _ in words[i+1:min(len(words), i+3)]]  # Get next 2 words
            
            # Get POS tag for the current word
            current_pos = self._get_word_pos(word, context)
            
            # Check for confusion set errors
            if processed_word in self.confusion_sets:
                # Get POS tags for potential confusion words
                confusion_words = self.confusion_sets[processed_word]
                for conf_word in confusion_words:
                    conf_pos = self._get_word_pos(conf_word, context)
                    # If the confusion word has a different POS tag that matches the context better
                    if conf_pos != current_pos and conf_pos in self.pos_confusion_sets.get(current_pos, set()):
                        errors.append((word, start, end, 'confusion_set'))
                        break
            
            # Check for real-word errors using both left and right context
            current_prob = self._get_word_probability(word, context, next_words)
            word_freq = self.word_frequencies.get(processed_word, 0)
            
            # Only flag as real-word error if both probability is low AND word is relatively rare
            if current_prob < self.real_word_threshold and word_freq < 1000:
                errors.append((word, start, end, 'real_word'))
        
        return errors
    
    def add_word(self, word: str) -> None:
        """Add a word to the dictionary."""
        processed_word = self._preprocess_word(word)
        if processed_word:
            self.dictionary.add(processed_word)
    
    def remove_word(self, word: str) -> None:
        """Remove a word from the dictionary."""
        processed_word = self._preprocess_word(word)
        self.dictionary.discard(processed_word)
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calculate the Levenshtein distance between two strings.
        The Levenshtein distance is the minimum number of single-character edits
        required to change one word into another.
        Substitutions count as 2 distance units, while insertions and deletions count as 1.
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                # Count substitutions as 2 distance units
                substitutions = previous_row[j] + (2 if c1 != c2 else 0)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def get_suggestions(self, word: str, context: List[str] = None, max_suggestions: int = 5, max_distance: int = 2) -> List[Tuple[str, int, float]]:
        """
        Generate spelling suggestions for a misspelled word using Levenshtein distance,
        n-gram probabilities, and confusion sets.
        """
        processed_word = self._preprocess_word(word)
        suggestions = []
        
        # Get POS tag for better suggestions
        current_pos = self._get_word_pos(word, context) if context else self._get_word_pos(word)
        
        # Check confusion sets first
        if processed_word in self.confusion_sets:
            confusion_words = self.confusion_sets[processed_word]
            for conf_word in confusion_words:
                conf_pos = self._get_word_pos(conf_word, context) if context else self._get_word_pos(conf_word)
                if conf_pos == current_pos or conf_pos in self.pos_confusion_sets.get(current_pos, set()):
                    suggestions.append((conf_word, 0, 1.0))  # High probability for confusion set words
        
        # Get Levenshtein distance-based suggestions
        for dict_word in self.dictionary:
            distance = self._levenshtein_distance(processed_word, dict_word)
            if 0 < distance <= max_distance:
                prob = self._get_word_probability(dict_word, context) if context else 0.0
                suggestions.append((dict_word, distance, prob))
        
        # Sort by combined score (lower distance and higher probability is better)
        suggestions.sort(key=lambda x: (x[1], -x[2]))
        return suggestions[:max_suggestions]

def main():
    # Example usage
    checker = SpellChecker()
    
    # Test text with some spelling errors and special characters
    text = "placbo likelihod treatd"
    errors = checker.check_text(text)
    
    print(f"Original text: {text}")
    print("\nSpelling errors found:")
    for word, start, end, error_type in errors:
        print(f"\nWord: '{word}' at position {start}-{end}, Error Type: {error_type}")
        suggestions = checker.get_suggestions(word)
        print("Suggestions:")
        for suggestion, distance, prob in suggestions:
            print(f"  - {suggestion} (distance: {distance}, probability: {prob})")

if __name__ == "__main__":
    main()
