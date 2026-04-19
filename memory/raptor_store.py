# Pseudo/design notes:
# This module implements RaptorStore and RaptorConfig which interacts with the RAPTOR library

from raptor import RetrievalAugmentation
import os
import pickle

class RaptorStore:
    def __init__(self):
        self.retrieval_aug = None

    def lazy_init(self):
        self.retrieval_aug = RetrievalAugmentation(api_key=os.getenv(cfg.openai_api_key_env))
        os.makedirs('raptor_dir', exist_ok=True)

    def ingest_texts(self, texts):
        # Safe adapter for adding documents/texts
        if hasattr(self.retrieval_aug, 'add_documents'):
            self.retrieval_aug.add_documents(texts)
        elif hasattr(self.retrieval_aug, 'add_texts'):
            self.retrieval_aug.add_texts(texts)
        elif hasattr(self.retrieval_aug, 'insert'):
            self.retrieval_aug.insert(texts)
        else:
            raise NotImplementedError("No suitable method found to ingest texts.")

    def query(self, question):
        # Safe adapter for querying
        if hasattr(self.retrieval_aug, 'answer_question'):
            return self.retrieval_aug.answer_question(question)
        elif hasattr(self.retrieval_aug, 'query'):
            return self.retrieval_aug.query(question)
        elif hasattr(self.retrieval_aug, 'ask'):
            return self.retrieval_aug.ask(question)
        else:
            raise NotImplementedError("No suitable method found to process query.")

    def save(self, path):
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.retrieval_aug, f)
        except Exception as e:
            print(f'Warning: Saving failed with error: {e}')

    def load(self, path):
        try:
            with open(path, 'rb') as f:
                self.retrieval_aug = pickle.load(f)
        except Exception as e:
            print(f'Warning: Loading failed with error: {e}')
            self.retrieval_aug = None
