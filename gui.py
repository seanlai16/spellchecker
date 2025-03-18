import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import messagebox
from spellchecker import SpellChecker

class WordListViewer(tk.Toplevel):
    def __init__(self, parent, words):
        super().__init__(parent)
        self.title("Word List Viewer")
        self.geometry("300x400")
        
        # Create main frame
        main_frame = ttk.Frame(self, padding="5")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create search frame
        search_frame = ttk.Frame(main_frame)
        search_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Create search entry
        self.search_var = tk.StringVar()
        self.search_var.trace_add('write', self.filter_words)
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Create word count label
        self.word_count_label = ttk.Label(search_frame, text="")
        self.word_count_label.grid(row=0, column=1, padx=5)
        
        # Create word list
        self.word_list = tk.Listbox(main_frame, width=30, height=15)
        self.word_list.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.word_list.yview)
        scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.word_list.configure(yscrollcommand=scrollbar.set)
        
        # Store all words
        self.all_words = sorted(words)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        search_frame.columnconfigure(0, weight=1)
        
        # Initial population of word list
        self.update_word_list(self.all_words)
    
    def filter_words(self, *args):
        """Filter words based on search text."""
        search_text = self.search_var.get().lower()
        if search_text:
            filtered_words = [word for word in self.all_words if search_text in word.lower()]
        else:
            filtered_words = self.all_words
        self.update_word_list(filtered_words)
    
    def update_word_list(self, words):
        """Update the word list with the given words."""
        self.word_list.delete(0, tk.END)
        for word in words:
            self.word_list.insert(tk.END, word)
        self.word_count_label.config(text=f"Words: {len(words)}")

class SpellCheckerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Spell Checker")
        self.root.geometry("500x350")
        
        # Initialize spell checker
        self.spell = SpellChecker()
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="5")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create text editor
        self.text_editor = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, width=45, height=12)
        self.text_editor.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure text tags for highlighting
        self.text_editor.tag_configure("non_word_error", background="pink")
        self.text_editor.tag_configure("real_word_error", background="yellow")
        
        # Create character count label
        self.char_count_label = ttk.Label(main_frame, text="Characters: 0/500")
        self.char_count_label.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(2, 0))
        
        # Create status label for checkmark and message
        self.status_label = ttk.Label(main_frame, text="✓ No errors found", foreground="green")
        self.status_label.grid(row=1, column=1, sticky=tk.E, pady=(2, 0))
        self.status_label.grid_remove()  # Hide initially
        
        # Create button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=5)
        
        # Create spell check button
        self.spell_check_button = ttk.Button(button_frame, text="Spell Check", command=self.run_spell_check)
        self.spell_check_button.pack(side=tk.LEFT, padx=5)
        
        # Create word list button
        self.word_list_button = ttk.Button(button_frame, text="View Word List", command=self.show_word_list)
        self.word_list_button.pack(side=tk.LEFT, padx=5)
        
        # Bind text change event
        self.text_editor.bind('<KeyRelease>', self.update_char_count)
        
        # Bind click event for suggestions
        self.text_editor.bind('<Button-1>', self.show_suggestions)
        
        # Store current errors for suggestions
        self.current_errors = []
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
    def update_char_count(self, event=None):
        current_text = self.text_editor.get("1.0", tk.END)
        char_count = len(current_text) - 1  # Subtract 1 for the newline character
        
        # Update character count label
        self.char_count_label.config(text=f"Characters: {char_count}/500")
        
        # If text exceeds 500 characters, remove the excess
        if char_count > 500:
            self.text_editor.delete("1.0", tk.END)
            self.text_editor.insert("1.0", current_text[:500])
            self.char_count_label.config(text="Characters: 500/500")
        
        # Remove existing highlights when text changes
        self.text_editor.tag_remove("non_word_error", "1.0", tk.END)
        self.text_editor.tag_remove("real_word_error", "1.0", tk.END)
    
    def highlight_errors(self, errors):
        """Highlight errors in the text editor using tags."""
        # Remove existing highlights
        self.text_editor.tag_remove("non_word_error", "1.0", tk.END)
        self.text_editor.tag_remove("real_word_error", "1.0", tk.END)
        
        # Add new highlights
        for word, start, end, error_type in errors:
            # Convert character positions to line.column format
            start_index = f"1.{start}"
            end_index = f"1.{end}"
            
            # Apply appropriate tag based on error type
            tag_name = "non_word_error" if error_type == 'non_word' else "real_word_error"
            self.text_editor.tag_add(tag_name, start_index, end_index)
    
    def show_word_list(self):
        """Open the word list viewer window."""
        WordListViewer(self.root, self.spell.dictionary)
    
    def run_spell_check(self):
        # Get text from editor
        text = self.text_editor.get("1.0", tk.END).strip()
        
        if not text:
            messagebox.showinfo("Empty Text", "Please enter some text to spell check.")
            self.status_label.grid_remove()  # Hide checkmark
            return
        
        # Check for spelling errors
        errors = self.spell.check_text(text)
        
        if not errors:
            self.status_label.grid()  # Show checkmark
        else:
            self.status_label.grid_remove()  # Hide checkmark
            # Highlight errors in the text editor
            self.highlight_errors(errors)

    def show_suggestions(self, event):
        """Show suggestions when clicking on a highlighted word."""
        # Get the index of the clicked position
        index = self.text_editor.index(f"@{event.x},{event.y}")
        
        # Check if the clicked position has any tags
        tags = self.text_editor.tag_names(index)
        
        # If no error tags, return
        if not any(tag in ["non_word_error", "real_word_error"] for tag in tags):
            return
        
        # Get the word at the clicked position
        word_start = int(index.split('.')[1])
        word_end = word_start
        
        # Find the word boundaries
        while word_start > 0 and self.text_editor.get(f"1.{word_start-1}").isalnum():
            word_start -= 1
        while word_end < len(self.text_editor.get("1.0", tk.END)) and self.text_editor.get(f"1.{word_end}").isalnum():
            word_end += 1
        
        # Get the word
        word = self.text_editor.get(f"1.{word_start}", f"1.{word_end}")
        
        # Get context for better suggestions
        context = self.text_editor.get(f"1.{max(0, word_start-50)}", f"1.{word_start}").split()[-2:]
        suggestions = self.spell.get_suggestions(word, context)
        
        if not suggestions:
            return
        
        # Create popup menu
        popup = tk.Menu(self.root, tearoff=0)
        
        # Add suggestions to menu
        for suggestion, distance, prob in suggestions:
            popup.add_command(
                label=f"{suggestion} (distance: {distance})",
                command=lambda s=suggestion: self.replace_word(word_start, word_end, s)
            )
        
        # Show popup at click position
        popup.tk_popup(event.x_root, event.y_root)

    def replace_word(self, start, end, new_word):
        """Replace the selected word with the chosen suggestion."""
        self.text_editor.delete(f"1.{start}", f"1.{end}")
        self.text_editor.insert(f"1.{start}", new_word)
        
        # Update highlights
        self.run_spell_check()

def main():
    root = tk.Tk()
    app = SpellCheckerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
