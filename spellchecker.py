import tkinter as tk
from tkinter import ttk
import re
import collections
from nltk.metrics.distance import edit_distance

class SpellingCheckerApp:
    def __init__(self, root, corpus_file):
        self.root = root
        self.root.title("Spelling Checker")
        
        self.word_counts = self.load_corpus(corpus_file)
        
        # Text Editor
        self.text_editor = tk.Text(root, height=10, width=50, wrap=tk.WORD)
        self.text_editor.pack(pady=10)
        
        # Check Button
        self.check_button = tk.Button(root, text="Check Spelling", command=self.check_spelling)
        self.check_button.pack()
        
        # Word List
        self.word_listbox = tk.Listbox(root, height=10, width=50)
        self.word_listbox.pack(pady=10)
        self.populate_word_list()
        
        # Suggestions
        self.suggestions_label = tk.Label(root, text="Suggested Corrections:")
        self.suggestions_label.pack()
        self.suggestions_listbox = tk.Listbox(root, height=5, width=50)
        self.suggestions_listbox.pack()
        
    def load_corpus(self, filename):
        with open(filename, encoding='utf-8') as f:
            words = re.findall(r'\w+', f.read().lower())
        return collections.Counter(words)
    
    def check_spelling(self):
        self.suggestions_listbox.delete(0, tk.END)
        text = self.text_editor.get("1.0", tk.END).lower()
        words = re.findall(r'\w+', text)
        
        self.text_editor.tag_remove("misspelled", "1.0", tk.END)
        
        for word in words:
            if word not in self.word_counts:
                self.highlight_word(word)
                self.suggest_corrections(word)
    
    def highlight_word(self, word):
        text = self.text_editor.get("1.0", tk.END)
        start = "1.0"
        while True:
            start = self.text_editor.search(word, start, stopindex=tk.END)
            if not start:
                break
            end = f"{start}+{len(word)}c"
            self.text_editor.tag_add("misspelled", start, end)
            self.text_editor.tag_config("misspelled", foreground="red")
            start = end
    
    def suggest_corrections(self, word):
        candidates = sorted(self.word_counts.keys(), key=lambda w: edit_distance(word, w))[:5]
        self.suggestions_listbox.insert(tk.END, f"{word} -> {', '.join(candidates)}")
    
    def populate_word_list(self):
        sorted_words = sorted(self.word_counts.keys())
        for word in sorted_words:
            self.word_listbox.insert(tk.END, word)

if __name__ == "__main__":
    corpus_file = 'corpus.txt'  # Example file
    root = tk.Tk()
    app = SpellingCheckerApp(root, corpus_file)
    root.mainloop()
