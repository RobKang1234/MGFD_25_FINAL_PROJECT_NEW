import typing
import os
import tempfile
import numpy as np 
import pandas as pd 
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import Counter, defaultdict
from gensim import corpora, models, similarities


class summarize_article():
    def remove_common_words(self):
        """Input must be a list of string
        [
            "Sentence 1",
            "Sentence 2"
            ...
        ]
        """
        stop_list = set('the a an to and of really'.split(" "))
        try:
            texts = [
                [word for word in document.lower().split() if word not in stop_list]
                for document in self
            ]
            return texts
        except TypeError as e:
            print("Type Error:", e)
        return None
    
    def get_frequency(self):
        """Get frequency dictionary 
        
        """
        frequency = defaultdict(int)
        
        for text in summarize_article.remove_common_words(self):
            for token in text:
                frequency[token]+=1
        
        return frequency
        
    def remove_word_only_appear_once_and_common_words(self):
        """
        
        """
        frequency = defaultdict(int)
        
        for text in summarize_article.remove_common_words(self):
            for token in text:
                frequency[token]+=1
      
        try:
            texts = [
                [token for token in char if frequency[token] > 1]
                for char in summarize_article.remove_common_words(self)
            ]
            return texts
        except TypeError as e:
            print("Type Error: ", e)
        return None
    
    def get_stat(self):
        """Get a summary statastic of the input article in the form of list
        
        """
        stop_list = set('the a an to and of really'.split(" "))
        dic = corpora.Dictionary(line.lower().split() for line in self)
        stop_ids = [
            dic.token2id[stopword]
            for stopword in stop_list
            if stopword in dic.token2id
        ]
        once_ids = [tokenid for tokenid, docfreq in dic.dfs.items() if docfreq == 1]
        dic.filter_tokens(stop_ids + once_ids)
        dic.compactify()
        
        return dic 
    
    def get_values_of_stat_dics(self):
        """Get all values from the stat dic, stat dic is the dic we can get from above
        
        Return a list of values!
        
        """
        return [self[key] for key in self]
       
    
    def lsi_corpus_model(self, num_topics, need_return:bool):
        """Input must go through frequency check, remove words 
        that only appear once and common words
        
        This function print to log/ Return None as default
        
        Print output to log as default no matter need_return == True || False
        """
        #Step 0: load if needed for reference for future

        #Step 1: Initilize
        dic = corpora.Dictionary(self)
        corpus = [dic.doc2bow(text) for text in self]
        tfidf = models.TfidfModel(corpus)
        #Step 2: Model Initials
        corpus_tfidf = tfidf[corpus]
        
        #Step 3: Build Latent Semantic Indexing into a letent 2d space
        lsi_model = models.LsiModel(corpus_tfidf, id2word=dic, num_topics=num_topics)
        corpus_lsi = lsi_model[corpus_tfidf]
        
        #Save for persistence
        try:
            with tempfile.NamedTemporaryFile(prefix='model-',suffix='.lsi', delete=False) as tmp:
                lsi_model.save(tmp.name)
        except:
            print("Model unable to save")
        
        if need_return==True:
            lsi_model.print_topics(num_topics)
            return corpus_lsi
        else:
            lsi_model.print_topics(num_topics)
            return None
        
    
    def similarity_queries(self, to_compare):
        """Input must go through frequency check, remove words 
        that only appear once and common words
        """
        #loading
        try:
            index = similarities.MatrixSimilarity.load('/tmp/matrix.index')
        except:
            print("Unsuccessfully Load, going to init")
            #Transformation
            dic = corpora.Dictionary(self)
            corpus = [dic.doc2bow(text) for text in self]
            #Define lsi, default topic is 2 
            lsi = models.LsiModel(corpus, id2word=dic, num_topics=2)
            index = similarities.MatrixSimilarity(lsi[corpus])
            
            #save index
            index.save('/tmp/matrix.index')
        
        dic = corpora.Dictionary(self)
        corpus = [dic.doc2bow(text) for text in self]
        #Define lsi, default topic is 2 
        lsi = models.LsiModel(corpus, id2word=dic, num_topics=2)
        vec_bow = dic.doc2bow(to_compare.lower().split())
        vec_lsi = lsi[vec_bow]
        
        sims = index[vec_lsi]
        # print (document_number, document_similarity) 2-tuples
        print("Documen Number, Document similarity")
        print(list(enumerate(sims)))
        
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        for doc_position, doc_score in sims:
            print(doc_score, documents[doc_position])
        


#Should be removed - Tested pass. 
if __name__ == "__main__":
    """
    money supply，money demand，
    fiscal policy，monetary policy，inflation，
    loan，consume，Bull market ，bear market，
    """
    documents = [
        "Cost of living is higher",    #Inflation 
        "Many people take loans", #Inceasing number of loans
        "The net export increases", #GDP high
        "People are optimistic about the economy", #
        "Can not afford the loans", #
        "Many people lose their jobs", #Employment
        "The investment amount increases",
        "Companies go bankrupt",
        "People save more money",
        "The Covid-19 becomes more serious"
       
        
    ]
    a = summarize_article.remove_word_only_appear_once_and_common_words(documents)
    print(a)
   
    df = pd.read_csv("/Users/robkang/Documents/MGFD_25_FINAL_CODE/credit.csv")
    body = df['body'].tolist()
    print("Titile Analysis")
    print("="*30)
    for c in df['title'].tolist():
        print("="*20)
        print(c)
        summarize_article.similarity_queries(a, str(c))
    print("Answer Analysis\n\n")
    print("="*30)
    for one in body:
        print(one)
        summarize_article.similarity_queries(a, str(one))
        print("\n\n")
    