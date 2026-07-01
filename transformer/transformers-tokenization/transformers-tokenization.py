import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        special_tokens=[
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]
        for token in special_tokens:
            self.word_to_id[token]=len(self.word_to_id)

        unique_words=set()
        
        for text in texts:
            unique_words.update(text.lower().split())
            
        for word in sorted(unique_words):
            self.word_to_id[word]=len(self.word_to_id)
            
        self.id_to_word={id:word for word,id in self.word_to_id.items()}
        self.vocab_size=len(self.word_to_id)
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE 
        token_ids = []
        for word in text.lower().split():
                if word in self.word_to_id:
                    token_ids.append(self.word_to_id[word])
                else:
                    token_ids.append(self.word_to_id[self.unk_token])
                    
        return token_ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        words=[]
        unk_id=self.word_to_id[self.unk_token]
        for token in ids:
            if token in self.id_to_word:
                 words.append(self.id_to_word[token])
            else:
                 words.append(self.id_to_word[unk_id])
                    
        return " ".join(words)
