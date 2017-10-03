# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 19:12:17 2017

@author: Kabir
"""

#Importing all important modules and methods required for our program
import numpy as np
from nltk.tokenize import sent_tokenize,word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import state_union
from nltk.tag import pos_tag
import string
from sklearn.cross_validation import train_test_split
from sklearn.multiclass import OneVsRestClassifier
import pyttsx
import speech_recognition as sr
from heapq import nlargest
from nltk.corpus import stopwords
from collections import defaultdict
import re

#This function trains all the questions of dataset using bernoulli naive bayes and decision trees
#to do sentiment analysis of the different kind of questions asked by user
def train_questions():
    #loading the text files containing questions
    txt = open("0.txt")
    questions0 = txt.read()
    txt.close()
    txt = open("1.txt")
    questions1 = txt.read()
    txt.close()
    txt = open("2.txt")
    questions2 = txt.read()
    txt.close()
    txt = open("3.txt")
    questions3 = txt.read()
    txt.close()
    #loading all the questions into a numpy array and creating a numpy array containing labels
    #of all the questions(to the class they belong)
    #Currently our program classifies baiscally 4 types of questions:
    #0 : if user wants to ask about a particular topic in the meet
    #1 : if user wants to know important topics discussed in the meet
    #2: if user wants to exit the program
    #3: if user wants the summary of the meet
    sentences0 = sent_tokenize(questions0)
    sentences1 = sent_tokenize(questions1)
    sentences2 = sent_tokenize(questions2)
    sentences3 = sent_tokenize(questions3)
    A = sentences0  + sentences1 + sentences2 + sentences3
    A = np.array(A)
    y = np.zeros(len(A))
    for i in range(len(sentences0),len(sentences1)+len(sentences0)):
        y[i] = 1
    for i in range(len(sentences1)+len(sentences0),len(sentences2)+len(sentences1)+len(sentences0)):
        y[i] = 2
    for i in range(len(sentences2)+len(sentences1)+len(sentences0),len(A)):
        y[i] = 3
    #Extracting features from our data using count vectorizer which just counts the occurences
    #of inidvidual words in the document. We can also use TFIDF but here it doesnt really matters
    #because size of the documents is relatively very small
    vectorizer = CountVectorizer(analyzer="word",stop_words="english")
    words = vectorizer.fit_transform(A)
    features = words.toarray()
    #Splitting the data into training and testing data
    features_train,features_test,y_train,y_test = train_test_split(features,y,test_size = 0.15, random_state = 42)
    #Since there are 4 classes to classify here we use OneVsAll classification technique
    #First we trained our model using Bernoulli Naive Bayes Classifier
    clf_bern = OneVsRestClassifier(BernoulliNB()).fit(features_train, y_train)
    #print clf_bern.score(features_test,y_test)
   #Gives an accuracy of 100% on test data
    
   #Then we used Decision Tree Classifier to train our model.
    clf_tree = OneVsRestClassifier(DecisionTreeClassifier(random_state=0)).fit(features_train, y_train)
    #print clf_tree.score(features_test,y_test)
    #Gives an accuracy of 88% on test data 
    return [clf_bern,clf_tree,vectorizer]

[clf_bern,clf_tree,vectorizer] = train_questions()

#this method classifies to which category a new question belongs
def classify_question(question,vectorizer = vectorizer, clf = clf_bern):
    b = vectorizer.transform([question])
    b = b.toarray()
    return clf.predict(b)[0]

#loading the data set of different minutes of the meets
lisa = state_union.fileids()
dataset = []
for ele in lisa:
    dataset.append(state_union.raw(ele))
for i in range(len(dataset)):
    dataset[i] = dataset[i].encode('utf-8')

#this funtion finds the most important words in the nth meet.
def important_words(n,dataset = dataset):
    data = dataset
    #removing punctuations and \n from the data
    for i in range(len(data)):
        data[i] = data[i].translate(None,string.punctuation)
    	data[i] = data[i].translate(None,"\n")
    
    # extracting features from data using TFIDF, here using TFIDF is preffered since documents
    # can be very large and there can be many irrelevant words which occur many times in the documents
    vectorizer = TfidfVectorizer(sublinear_tf=True,max_features=500,stop_words='english')
    words = vectorizer.fit_transform(data)
    X = words.toarray()
    
    #getting names of the features found by tfidf vectorizer
    feature_array = np.array(vectorizer.get_feature_names())
    
    #getting top m words with highest tfidf
    m = 5
    top_m = X[n].argsort()[-1*m:][::-1]
    imp_words = []
    for top in top_m:
        imp_words.append(feature_array[top])
    
    #top_n = feature_array[tfidf_sorting][:n]
    for i in range(len(imp_words)):
        imp_words[i] = str(imp_words[i])
    return imp_words

# this method finds information about a particular topic as asked by the user in the document
def query(question,n,dataset=dataset):
    
    #finding the topic asked by the user from the question by finding nouns in the question
    tagged_sent = pos_tag(question.split())
    nouns = [word for word,pos in tagged_sent if pos == 'NN']
    word = nouns[0]
    #tokenizing the document to sentences and looking for the sentences containing our word
    sentences = sent_tokenize(dataset[n])
    info = ""
    for i in range(len(sentences)):
        words = sentences[i].split()
        if word in words:
            if i>0 and i < len(sentences)-1:
                info = info + " " + sentences[i-1] + sentences[i] + " " +sentences[i+1] + "\n"
            else:
                info = info + " " + sentences[i]
    return [info,word]

#next 2 methods are used in finding the summary of the document
#defining stop words as the ones provided by nltk corpus and the punctuations
stop_words = set(stopwords.words('english') + list(string.punctuation))

#this method finds frequency of all the words in a list of list of words of sentences of the documents
def calculate_frequency(words_list,stop_words = stop_words):
    #initializing a dictionary which will map word to its frequency in the document
    frequency = defaultdict(int)
    
    #defined upper limit and lower limit for the frequency, too high frequency means
    # that the word is most likely to be a stop word and too low frequency means
    #its not very relevant as it occurs very few times in the document
    upper = 0.9
    lower = 0.1
    
    #finding frequency of all non stop words by finding number of times they occur in the document
    for words in words_list:
        for word in words:
            if word not in stop_words:
                frequency[word] +=1

    #normalising the frequencies and deleting those words from dictionary which have frequencies
    #lower than the lower bound or higher than the upper bound
    m = float(max(frequency.values()))
    for word in frequency.keys():
        frequency[word] = frequency[word]/m
        if frequency[word] > upper or frequency[word] < lower:
            del frequency[word]
    return frequency

#This function finds the summary of the document by extracting top n sentences having the words with
#highest frequencies
def summarize(text,n=5):
    #tokenizing the document to sentences
    sentences = sent_tokenize(text)
    if n > len(sentences):
        n = len(sentences)/2

    #creating a word_list containing a list of list of words of the sentences of the douments
    words_list = [word_tokenize(sentence.lower()) for sentence in sentences]
    frequency = calculate_frequency(words_list)
    
    #creating a rank dictionary which maps a sentence to the sum of frequencies of words contained in it
    ranks = defaultdict(int)
    for i,words in enumerate(words_list):
        for word in words:
            if word in frequency:
                ranks[i] += frequency[word]
    
    #finding summary by finding the sentences with highest key values in the rank dictionary
    sent_ind =nlargest(n, ranks, key=ranks.get)
    summary = []
    for ind in sent_ind:
        summary.append(sentences[ind])
    return summary

#Now all the necessary methods have been defined, now we will write our main program

#initializing an engine which converts text to speech and changing properties like
# rate and the voice      
engine = pyttsx.init()
voices = engine.getProperty('voices')
engine.setProperty('rate', 120)
engine.setProperty('voice', voices[1].id)

#the name of the user will be stored in name, its default value has been set to arnold
name = "Arnold"
print "Hii!!! I am dolores.\nSome people choose to see the ugliness in this world, the disarray. I choose to see the beauty. \
To believe there is an order to our days. A purpose. What is your good name"
   
engine.say( "Hii!!! I am dolores.\nSome people choose to see the ugliness in this world, the disarray. I choose to see the beauty. \
To believe there is an order to our days. A purpose. What is your good name")

engine.runAndWait()

# now initializing the speech to text converter which listens to the input given
#by the user through the microphone and then converts it into text.
r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)
 
# Speech recognition using Google Speech Recognition
try:
    # for testing purposes, we're just using the default API key
    # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
    # instead of `r.recognize_google(audio)`
    name =  r.recognize_google(audio)
    print "You said: " + name
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))

print "Hii "+name+"! How may I help you?"
engine.say("Hii "+name+"! How may I help you?")
engine.runAndWait()

while True:
    
    #Asking the question
    #question can be in the format like, what important things were discussed
    #in the meet 5, what was discussed about democracy in the meet 3,
    #give me summary of the meet 4 or just simply goodbye to exit
    question = ""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)
    
    try:
        question =  r.recognize_google(audio)
        print "You said : " + question
        
    except sr.UnknownValueError:
        question = "goodbye dolores"
    except sr.RequestError as e:
        question = "goodbye dolores"
    
        
    #finding out the meet asked by the user
    question = question.lower()
    num = re.findall('\d+', question)
    if(len(num)==0):
        n=0
    else:
        n = int(num[0])
    if n > len(dataset):
        n = len(dataset)-1
    #running our classify_question method through which our program figures 
    #out what is asked by the user and according give the result desired by
    #the user
    
    #if classif_question returns 2 it means user wants to exit
    if classify_question(question) == 2:
        engine.say("Goodbye "+ name +" Have a good day")
        engine.runAndWait()
        break
    
    #if classify_question returns 1 it means user wants to know the important
    #things discussed in the meet
    elif classify_question(question) == 1:
      #found the important words using important_words method  
    	top_n = important_words(n)    
     	engine.say("The important topics discussed in the meet were: ")
    	engine.runAndWait()
    	for top in top_n :
    		engine.say(top)
    		engine.runAndWait()
    		print top
    
    #if classify_question returns 0 it means user wants to know about a particular
    #topic in the document
    elif classify_question(question) == 0:
        
        #find the information about that word using our query function
        [info,word] = query(question,n)
        print "Would You like me to read the lines or just print them on the screen"
        engine.say("Would You like me to read the lines or just print them on the screen")
        engine.runAndWait()
        response = ""
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Say Something")
            audio = r.listen(source)
        try:
            response = r.recognize_google(audio)
            print "You said: " + response
        except sr.UnknownValueError:
            response = "print"
        except sr.RequestError as e:
            response = "print"
    	
        response = response.lower()
        response_words = response.split()
        if "read" in response_words:
            if info == "":
                engine.say("I am afraid nothing was discussed about "+ word +" in the meet")
                engine.runAndWait()
            else:
                print info
                engine.say(info)
                engine.runAndWait()
        else:
            if info == "":
                print("I am afraid nothing was discussed about "+ word +" in the meet")
            else:
                print(info)
                engine.say("Press enter when you have finished reading.")
                engine.runAndWait()
                wait = raw_input("Press enter when you have finished reading.")

    # if classif_question returns 3 it means user wants to know summary of the
    #the meet        
    if classify_question(question) == 3:
        summary = summarize(dataset[n])
        print "Okay i will summarize the meet for you"
        engine.say("Okay i will summarize the meet for you")
        engine.runAndWait()
        print summary
        engine.say(summary)
        engine.runAndWait()
       
    engine.say("What else can i do for you " + name)
    engine.runAndWait()