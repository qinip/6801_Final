import re
import csv
import time
from gensim.parsing.preprocessing import * #provides a number of convenience preprocessing functions optimized for speed
from gensim.models.word2vec import Word2Vec
import gensim
from multiprocessing import cpu_count
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import KeyedVectors
import os




def preprocessing(file):

    CUSTOM_STOPWORDS = """
    a about above across after afterwards again against all almost alone along already also although always am among amongst amoungst amount an and another any anyhow anyone anything anyway anywhere are around as at back be
    became because become becomes becoming been before beforehand behind being below beside besides between beyond both bottom but by call can
    cannot cant co con could couldnt cry de describe
    detail did didn do does doesn doing don done down due during
    each eg eight either eleven else elsewhere empty enough etc even ever every everyone everything everywhere except few fifteen
    fify fill find fire first five for former formerly forty found four from front full further 
    get give go
    had has hasnt have hence here hereafter hereby herein hereupon how however hundred 
    i ie if in inc indeed interest into is it its itself 
    keep last latter latterly least less ltd
    just 
    kg km 
    made make many may me meanwhile might mill mine more moreover most mostly move much must my myself name namely
    neither never nevertheless next nine no nobody none noone nor not nothing now nowhere of off
    often on once one only onto or other others otherwise our ours ourselves out over own part per
    perhaps please put rather re
    quite
    rather really regarding
    same say see seem seemed seeming seems serious several should show side since sincere six sixty so some somehow someone something sometime sometimes somewhere still such system 
    th take ten than that the their them themselves then thence there thereafter thereby therefore therein thereupon these they thick thin third this those though three through throughout thru thus to together too top toward towards twelve twenty two 
    un under until up unless upon us used using
    various very very via
    was we well were what whatever when whence whenever where whereafter whereas whereby wherein whereupon wherever whether which while whither who whoever whole whom whose why will with within without would yet you
    your yours yourself yourselves
    """    
    STOPWORDS = frozenset(w for w in CUSTOM_STOPWORDS.split() if w)

    def remove_stopwords_customized(s):
        s = utils.to_unicode(s)
        return " ".join(w for w in s.split() if w not in STOPWORDS)    

    CUSTOM_FILTERS = [lambda x: x.lower(), #To lowercase
                      lambda text: re.sub(r'https?:\/\/.*\s', '', text, flags=re.MULTILINE), #To Strip away URLs
                      #split_alphanum, #Add spaces between digits & letters in s using RE_AL_NUM.
                      strip_tags, #Remove tags from s using RE_TAGS.
                      strip_non_alphanum,#Remove non-alphabetic characters from s using RE_NONALPHA.
                      strip_punctuation, #Replace punctuation characters with spaces in s using RE_PUNCT.
                      strip_numeric, #Remove digits from s using RE_NUMERIC.
                      strip_multiple_whitespaces,#Remove repeating whitespace characters (spaces, tabs, line breaks) from s and turns tabs & line breaks into spaces using RE_WHITESPACE.
                      #remove_stopwords, # Set of 339 stopwords from Stone, Denis, Kwantes (2010).
                      remove_stopwords_customized, # Set of stopwords, modified from Stone, Denis, Kwantes (2010) to not eliminate gender words such as he, she, him, etc
                      #lambda x:" ".join(w for w in x.split() if w not in stopword_file) #Custom stopwords
                      lambda x: strip_short(x, minsize=2), #Remove words with length lesser than minsize from s. reduced from 3 to 2 in order to incorporate he
                      #stem_text #Transform s into lowercase and stem it.
                     ]

    tic = time.time() # Start timing


    csv.field_size_limit(2147483647)
    with open(file,'r', newline='',encoding="utf-8") as inpFile:

        csvObject = csv.reader(inpFile, delimiter=',',quotechar='"')

        wordThreshold=5 #Important: filter out sentences with less than wordThreshold words

        sentences = []
        for csvEntry in csvObject:
            if len(csvEntry)>1:
                lines = csvEntry[1].split('\n') #csvEntry[0] is url csvEntry[1] is text Fetched from URL

                for line in lines: #Different elements appear in their own line
                    words = preprocess_string(line,CUSTOM_FILTERS)

                    if len(words)>wordThreshold: #Important: filter out sentences with less than wordThreshold words
                        sentences.append(words)

    toc = time.time() # Start timing
    computationTime = toc-tic

    print("Reading Corpus file and preprocessing time:" +str(computationTime)+" seconds")

    print(" printing Top 2 and last sentences For sanity check")
    for i, s in enumerate(sentences[0:2]):
        print(i,s)
    print(len(sentences),sentences[-1])

    print(" stats about Corpus read from file")
    wordsInCorpus = 0
    for i, s in enumerate(sentences):
        wordsInCorpus += len(s)
    print("Number of words in corpus:",wordsInCorpus)
    print("Number of sentences in corpus:",len(sentences))
    #for i, s in enumerate(sentences[0:30]):
    #    print(i,s)
    return sentences

def training(sentences):
    #Training the model
    tic = time.time() # Start timing

    model = Word2Vec(sentences, # The sentences iterable can be simply a list of lists of tokens, but for larger corpora, consider an iterable that streams the sentences directly from disk/network
             sg=1, #Defines the training algorithm. If 1, skip-gram is employed; otherwise, CBOW is used
             size=300,#Dimensionality of the feature vectors
             window=10,# The maximum distance between the current and predicted word within a sentence
             min_count=5, #Ignores all words with total frequency lower than this
             workers= 10, #cpu_count()-12, #Use these many worker threads to train the model (=faster #training with multicore machines).
             hs = 0, # int {1,0}) – If 1, hierarchical softmax will be used for model training. If set to 0, and negative is non-zero, negative sampling will be used.
             negative = 10, # If > 0, negative sampling will be used, specifies how many “noise words” should be drawn (usually between 5-20). If set to 0, no negative sampling is used.
             sample = 0.0001, # The threshold for configuring which higher-frequency words are randomly downsampled, useful range is (0, 1e-5). #subsampling is a method of diluting very frequent words, akin to removing stop-words. The subsampling method presented in (Mikolov et al., 2013) randomly removes words that are more frequent than some threshold t with a probability of p, where f marks the word’s corpus frequency        
             iter = 5, # (int) – Number of iterations (epochs) over the corpus. //5
             max_final_vocab = 50000, # (int, optional) – Limits the vocab to a target vocab size by automatically picking a matching min_count. If the specified min_count is more than the calculated min_count, the specified min_count will be used. Set to None if not required.
             #max_vocab_size = 100000, # (int, optional) – Limits the RAM during vocabulary building; if there are more unique words than this, then prune the infrequent ones. Every 10 million word types need about 1GB of RAM. Set to None for no limit.
             #callbacks=[epoch_evaluation], # Sequence of callbacks to be executed at specific stages during training
             #compute_loss = True,# computes and stores loss value which can be retrieved using get_latest_training_loss().
             #ns_exponent = 0.5, # The exponent used to shape the negative sampling distribution. A value of 1.0 samples exactly in proportion to the frequencies, 0.0 samples all words equally, while a negative value samples low-frequency words more than high-frequency words. The popular default value of 0.75 was chosen by the original Word2Vec paper
             #cbow_mean = 0, # If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
             #min_alpha = float # Learning rate will linearly drop to min_alpha as training progresses.                     
            )

    toc = time.time() # Start timing
    computationTime = toc-tic
    print("Computing time for training the model:" +str(computationTime)+" seconds")
    wordsInCorpus = sum([len(l) for l in sentences])
    print("Number of words processed per second:",wordsInCorpus/computationTime)
    print(model)
    print("Most frequent words In model: ", model.wv.index2word[:10])

    return model

class EpochSaver(CallbackAny2Vec):
    '''Callback to save model after each epoch.'''
    def __init__(self,uni,outputPath):
        self.epoch = 1
		
        self.uni=uni
        self.outputPath = outputPath
        self.pearsonCorrelationTimeSeries = []
        self.spearmanCorrelationTimeSeries = []
        self.pearsonCorrelationMENTimeSeries = []
        self.pearsonCorrelationSimlexTimeSeries = []
        self.averageSemanticAccuracyTimeSeries = []
        self.averageSyntacticAccuracyTimeSeries = []
        self.averageTotalAccuracyTimeSeries = []
        
        self.averageBATS1InflectionalMorphologyTimeSeries = []
        self.averageBATS2DerivationalMorphologyTimeSeries = []
        self.averageBATS3EncyclopedicSemanticsTimeSeries = []
        self.averageBATS4LexicographicSemanticsTimeSeries = []
        
        
    def on_epoch_end(self, model):
        self.training_loss = model.get_latest_training_loss()
        print("Training Loss: ",self.training_loss)        
        self.name = self.uni + 'Epoch' + str(self.epoch)
        self.outputFilePath = os.path.join(self.outputPath, self.name)

		#saving model after each epoch
        model.save(self.outputFilePath) 
        
        # for some unknown reason the call to the accuracy function always operates on the epoch 1 model
        # as a way around, I save the current model, load it and call the accuracy function for the temporal model loaded from disk
        self.modelTemp = KeyedVectors.load(self.outputFilePath) 
        self.pearsonCorrelation,self.spearmanCorrelation,self.pearsonCorrelationMEN,self.pearsonCorrelationSimlex,self.averageSemanticAccuracy,self.averageSyntacticAccuracy,self.averageTotalAccuracy,self.averageBATS1InflectionalMorphologyAccuracy,self.averageBATS2DerivationalMorphologyAccuracy,self.averageBATS3EncyclopedicSemanticsAccuracy,self.averageBATS4LexicographicSemanticsAccuracy = evaluation(self.modelTemp)        
        del(self.modelTemp)

        self.pearsonCorrelationTimeSeries.append(round(self.pearsonCorrelation,2))
        self.spearmanCorrelationTimeSeries.append(round(self.spearmanCorrelation,2))
        self.pearsonCorrelationMENTimeSeries.append(round(self.pearsonCorrelationMEN,2))
        self.pearsonCorrelationSimlexTimeSeries.append(round(self.pearsonCorrelationSimlex,2))
        self.averageSemanticAccuracyTimeSeries.append(round(self.averageSemanticAccuracy,2))
        self.averageSyntacticAccuracyTimeSeries.append(round(self.averageSyntacticAccuracy,2))
        self.averageTotalAccuracyTimeSeries.append(round(self.averageTotalAccuracy,2))
        self.averageBATS1InflectionalMorphologyTimeSeries.append(round(self.averageBATS1InflectionalMorphologyAccuracy,2))
        self.averageBATS2DerivationalMorphologyTimeSeries.append(round(self.averageBATS2DerivationalMorphologyAccuracy,2))
        self.averageBATS3EncyclopedicSemanticsTimeSeries.append(round(self.averageBATS3EncyclopedicSemanticsAccuracy,2))
        self.averageBATS4LexicographicSemanticsTimeSeries.append(round(self.averageBATS4LexicographicSemanticsAccuracy,2))
        
        print("average Pearson Correlation: ",self.pearsonCorrelation)    
        print("average Spearman Correlation: ",self.spearmanCorrelation)  
        print("average Pearson MEN Correlation: ",self.pearsonCorrelationMEN)
        print("average Pearson Simlex Correlation: ",self.pearsonCorrelationSimlex)
        print("averageSemanticAccuracy: ",self.averageSemanticAccuracy)
        print("averageSyntacticAccuracy: ",self.averageSyntacticAccuracy)
        print("averageTotalAccuracy: ",self.averageTotalAccuracy)
        print("averageBATS1InflectionalMorphologyAccuracy: ",self.averageBATS1InflectionalMorphologyAccuracy)
        print("averageBATS2DerivationalMorphologyAccuracy: ",self.averageBATS2DerivationalMorphologyAccuracy)
        print("averageBATS3EncyclopedicSemanticsAccuracy: ",self.averageBATS3EncyclopedicSemanticsAccuracy)
        print("averageBATS4LexicographicSemanticsAccuracy: ",self.averageBATS4LexicographicSemanticsAccuracy)

        print("\n***********\n")

        print("Time Series Pearson Correlation: ",self.pearsonCorrelationTimeSeries)    
        print("Time Series Spearman Correlation: ",self.spearmanCorrelationTimeSeries)
        print("Time Series Pearson MEN Correlation: ",self.pearsonCorrelationMENTimeSeries) 
        print("Time Series Pearson Simlex Correlation: ",self.pearsonCorrelationSimlexTimeSeries) 
        print("Time Series averageSemanticAccuracy: ",self.averageSemanticAccuracyTimeSeries)
        print("Time Series averageSyntacticAccuracy: ",self.averageSyntacticAccuracyTimeSeries)
        print("Time Series averageTotalAccuracy: ",self.averageTotalAccuracyTimeSeries)
        print("Time Series averageBATS1InflectionalMorphologyAccuracy: ",self.averageBATS1InflectionalMorphologyTimeSeries) 
        print("Time Series averageBATS2DerivationalMorphologyAccuracy: ",self.averageBATS2DerivationalMorphologyTimeSeries)
        print("Time Series averageBATS3EncyclopedicSemanticsAccuracy: ",self.averageBATS3EncyclopedicSemanticsTimeSeries)
        print("Time Series averageBATS4LexicographicSemanticsAccuracy: ",self.averageBATS4LexicographicSemanticsTimeSeries)

        self.epoch += 1

     
	

def evaluation(model,restrict_vocabSimilarity=50000,restrict_vocabAnalogies=30000):
    # Evaluation WordSim-353
    ewp = model.wv.evaluate_word_pairs('./evaluation/wordsim353.tsv', restrict_vocab=restrict_vocabSimilarity) #50000
    #ewp = model.wv.evaluate_word_pairs('wordsim353.tsv')
    pearsonCorrelation = ewp[0][0]
    spearmanCorrelation = ewp[1][0]
    print("similarity WordSim-353:", ewp)  
    
    # Evaluation MEN 
    ewpMEN = model.wv.evaluate_word_pairs('./evaluation/MEN_dataset_natural_form_fullTabSeparated', restrict_vocab=restrict_vocabSimilarity) #50000
    #ewp = model.wv.evaluate_word_pairs('wordsim353.tsv')
    pearsonCorrelationMEN = ewpMEN[0][0]
    spearmanCorrelationMEN = ewpMEN[1][0]
    print("similarity MEN-3000:", ewpMEN)  
    
    # Evaluation Simlex-999 
    ewpSimlex = model.wv.evaluate_word_pairs('./evaluation/SimLex-999TabSeparated.txt', restrict_vocab=restrict_vocabSimilarity) #50000
    #ewp = model.wv.evaluate_word_pairs('wordsim353.tsv')
    pearsonCorrelationSimlex = ewpSimlex[0][0]
    spearmanCorrelationSimlex = ewpSimlex[1][0]
    print("similarity Simlex-999:", ewpSimlex)  
    
    # Analogies
    #r = model.wv.accuracy('questions-words.txt', restrict_vocab=30000) #30,000
    r = model.wv.evaluate_word_analogies('./evaluation/questions-words.txt', restrict_vocab=restrict_vocabAnalogies,case_insensitive=True,dummy4unknown=False) #30,000
    #r = model.wv.accuracy('questions-words.txt')
    correct = 0
    incorrect = 0
    for ir in r[1][:5]:
        correct += len(ir['correct'])
        incorrect += len(ir['incorrect'])
#         print(len(ir['correct']),len(ir['incorrect']),len(ir['incorrect'])+len(ir['correct']))    
    try:      
#         print(correct)
#         print(incorrect+correct)        
        averageSemanticAccuracy = correct/(correct+incorrect) 
    except Exception as error:
        averageSemanticAccuracy = 0
#     print("--")        
    correct = 0
    incorrect = 0
    for ir in r[1][5:-1]:
        correct += len(ir['correct'])
        incorrect += len(ir['incorrect'])
        #print(len(ir['correct']),len(ir['incorrect']),len(ir['incorrect'])+len(ir['correct']))    
    try:
#         print(correct)
#         print(incorrect+correct)
        averageSyntacticAccuracy = correct/(correct+incorrect)     
    except ZeroDivisionError as error:
        averageSyntacticAccuracy = 0
    
    averageTotalAccuracy = len(r[1][14]['correct'])/(len(r[1][14]['incorrect'])+len(r[1][14]['correct']))
    
   # BATS Analogies
    #r = model.wv.accuracy('questions-words.txt', restrict_vocab=30000) #30,000
    r = model.wv.evaluate_word_analogies('./evaluation/1_Inflectional_morphology.txt', restrict_vocab=restrict_vocabAnalogies,case_insensitive=True,dummy4unknown=False) #30,000
    indexTotal = len(r[1]) - 1
    averageBATS1InflectionalMorphologyAccuracy = len(r[1][indexTotal]['correct'])/(len(r[1][indexTotal]['incorrect'])+len(r[1][indexTotal]['correct']))

    r = model.wv.evaluate_word_analogies('./evaluation/2_Derivational_morphology.txt', restrict_vocab=restrict_vocabAnalogies,case_insensitive=True,dummy4unknown=False) #30,000
    indexTotal = len(r[1]) - 1
    averageBATS2DerivationalMorphologyAccuracy = len(r[1][indexTotal]['correct'])/(len(r[1][indexTotal]['incorrect'])+len(r[1][indexTotal]['correct']))

    r = model.wv.evaluate_word_analogies('./evaluation/3_Encyclopedic_semantics.txt', restrict_vocab=restrict_vocabAnalogies, case_insensitive=True,dummy4unknown=False) #30,000
    indexTotal = len(r[1]) - 1
    averageBATS3EncyclopedicSemanticsAccuracy = len(r[1][indexTotal]['correct'])/(len(r[1][indexTotal]['incorrect'])+len(r[1][indexTotal]['correct']))

    r = model.wv.evaluate_word_analogies('./evaluation/4_Lexicographic_semantics.txt', restrict_vocab=restrict_vocabAnalogies,case_insensitive=True,dummy4unknown=False) #30,000
    indexTotal = len(r[1]) - 1
    averageBATS4LexicographicSemanticsAccuracy = len(r[1][indexTotal]['correct'])/(len(r[1][indexTotal]['incorrect'])+len(r[1][indexTotal]['correct']))
    
    return [pearsonCorrelation,spearmanCorrelation,pearsonCorrelationMEN, pearsonCorrelationSimlex,averageSemanticAccuracy,averageSyntacticAccuracy,averageTotalAccuracy,averageBATS1InflectionalMorphologyAccuracy,averageBATS2DerivationalMorphologyAccuracy,averageBATS3EncyclopedicSemanticsAccuracy,averageBATS4LexicographicSemanticsAccuracy]
