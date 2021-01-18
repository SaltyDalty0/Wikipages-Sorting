#!/usr/bin/env python3
#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 04:07:23 2020

@author: DaltonGlove

Credit to Github creators and content: SamujjalSarma/Text-Classification-using-ML-algorithms.
    Ifiokcharles/Natural-language-Processing
    panyang/Wikipedia_Word2vec
    chrisjmccormick/LSA_Classification
    
    doc2vec & word2vec documentation and classification as detailed
    in Distributed Representations of Sentences and Documents,
    url: https://arxiv.org/pdf/1405.4053v2.pdf
    
    also Joseph Wilk's blog/website http://blog.josephwilk.net/
"""

#Defining function for cleaning text
def cleanText(text):
    '''text = re.sub('[^ a-zA-Z]','',text)
    text = re.sub(r' +', ' ', text)'''
    # Remove all the special characters
    text = text.replace(',', ' ')
    text = text.replace('.', ' ')
    text = text.replace('[', ' ')
    text = text.replace(']', ' ')
    text = text.replace('(', ' ')
    text = text.replace(')', ' ')
    for item in ('0','1','2','3','4','5','6','7','8','9'):
        text = text.replace(item, ' ')
    text = text.replace('"', ' ')
    text = re.sub(r'\W', ' ', str(text))
    #document = re.sub('\.|\!|\?', ' ', document)
    # remove all single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # Remove single characters from the start
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)     
    # Remove numbers characters from the start
    text = re.sub(r'[0123456789]', ' ', text)
    # Substituting multiple spaces with single space
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    # Removing prefixed 'b'
    text = re.sub(r'^b\s+', '', text)
    # Converting to Lowercase
    text = text.lower()
    return text


#attempt different methods for distance measurements, cosine similarity and euclidian distance

import math

def counter_cosine_similarity(c1, c2):
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    return dotprod / (magA * magB)

def length_similarity(c1, c2):
    lenc1 = sum(c1.values())
    lenc2 = sum(c2.values())
    return min(lenc1, lenc2) / float(max(lenc1, lenc2))
    
def similarity_score(c1, c2):
    #c1, c2 = Counter(l1), Counter(l2)
    return length_similarity(c1, c2) * counter_cosine_similarity(c1, c2)


#Implement two versions of knn algorithm from homework adjusted
#for cosine similarity and euclidean distance measures
#unused since sklearns algorithm is most efficient


def k_near_pluscount(wiki_pages, test_page, k):

    #cosine similarity method

    dist_list=[]
    for i in range(len(wiki_pages)):
        dist = similarity_score(test_page, wiki_pages[i])     #np.linalg.norm(test_page - wiki_pages[i])
        #print(dist)
        dist_list+=[[dist, i]]
        
    dist_list.sort(key=lambda tup: tup[0])
    
    neighbor_pts=[]
    neighbor_labels=[]
    for i in range(k):
    
        neighbor_pts+=[wiki_pages[int(dist_list[i][1])]]
        neighbor_labels+=[y_pages[int(dist_list[i][1])]]  #change to right column of dataframe
    return neighbor_pts, neighbor_labels 

def k_near_pluswords(wiki_pages, test_page, k):

    #finds euclidian distance between vectors
    
    dist_list=[]
    counters=[]
    for page in wiki_pages:
        counters+=[Counter(page.split())]
    vocab = test_page.split()
    test_count = Counter(vocab)
    i=0
    for page in counters:

        dist = math.sqrt(sum((test_count.get(k, 0) - page.get(k, 0))**2 for k in set(test_count.keys()).union(set(page.keys()))))

        dist_list+=[[dist, i]]
        i+=1
    
    dist_list.sort(key=lambda tup: tup[0])
    
    neighbor_pts=[]
    neighbor_labels=[]
    for i in range(k):
    
        neighbor_pts+=[wiki_pages[int(dist_list[i][1])]]
        neighbor_labels+=[y_pages[int(dist_list[i][1])]]  #change to right column of dataframe
    return neighbor_pts, neighbor_labels

def NN_wordsclassifier(wiki_pages, test_page, k, true_target_name):

    #one pt at a time use euclidean distance
    neighbors, classify = k_near_pluswords(wiki_pages, test_page, k)[0], k_near_pluswords(wiki_pages, test_page, k)[1]
    
    Techclass=0
    Filmsclass=0
    Healthclass=0
    for i in range(len(classify)):
        if classify[i]=="Tech":
            Techclass+=1
        if classify[i]=="Films":
            Filmsclass+=1
    if Techclass==Filmsclass:
        coinflip = np.random.binomial(1, .5, 1)
        if int(coinflip)==1: ret = "Tech"
        else: ret = "Films"
    if Techclass==Healthclass:
        coinflip = np.random.binomial(1, .5, 1)
        if int(coinflip)==1: ret = "Tech"
        else: ret = "Health"
    if Filmsclass==Healthclass:
        coinflip = np.random.binomial(1, .5, 1)
        if int(coinflip)==1: ret = "Films"
        else: ret = "Health"
    if Techclass>Filmsclass and Techclass>Healthclass: ret = "Tech"
    if Filmsclass>Techclass and Filmsclass>Healthclass: ret = "Films"
    if Healthclass>Filmsclass and Healthclass>Techclass: ret = "Health"
    return ret

def NN_countclassifier(wiki_pages, test_page, k, true_target_name):
    
    #one pt at a time use cosine similarity score
    neighbors, classify = k_near_pluscount(wiki_pages, test_page, k)[0], k_near_pluscount(wiki_pages, test_page, k)[1]
    
    #implement algorithm for classification
    
    Techclass=0
    Filmsclass=0
    Healthclass=0
    for i in range(len(classify)):
        if classify[i]=="Tech":
            Techclass+=1
        if classify[i]=="Films":
            Filmsclass+=1
            
        if classify[i]=="Health":
            Healthclass+=1
            
    if Techclass==Filmsclass:
        coinflip = np.random.binomial(1, .5, 1)
        if int(coinflip)==1: ret = "Tech"
        else: ret = "Films"
    if Techclass==Healthclass:
        coinflip = np.random.binomial(1, .5, 1)
        if int(coinflip)==1: ret = "Tech"
        else: ret = "Health"
    if Filmsclass==Healthclass:
        coinflip = np.random.binomial(1, .5, 1)
        if int(coinflip)==1: ret = "Films"
        else: ret = "Health"
    if Techclass>Filmsclass and Techclass>Healthclass: ret = "Tech"
    if Filmsclass>Techclass and Filmsclass>Healthclass: ret = "Films"
    if Healthclass>Filmsclass and Healthclass>Techclass: ret = "Health"
    return ret

#import necessary pkgs and testdata

import numpy as np
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import requests
import bs4
import pandas as pd
import collections, re
from sklearn.decomposition import LatentDirichletAllocation,PCA
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer, TfidfTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import nltk
from nltk import FreqDist
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Getting rid of unnecessary warnings
import warnings; 
warnings.simplefilter('ignore')


#initialize urls -> adjust into text file to be further added to

"""Check Wikipedia's list of lists of lists for the more urls"""

#Cultural URLs
film_url = 'https://en.wikipedia.org/wiki/Film'
bolly_url = 'https://en.wikipedia.org/wiki/Bollywood'
cineindia_url = 'https://en.wikipedia.org/wiki/Cinema_of_India'
holly_url = 'https://en.wikipedia.org/wiki/Hollywood'
cineus_url = 'https://en.wikipedia.org/wiki/Cinema_of_the_United_States'
filmhist_url = 'https://en.wikipedia.org/wiki/History_of_film'
movietheater_url = 'https://en.wikipedia.org/wiki/Movie_theater'
performarts_url = 'https://en.wikipedia.org/wiki/Performing_arts'
blockbuster_url = 'https://en.wikipedia.org/wiki/Blockbuster_(entertainment)#Blockbuster_films'
theatre_url = 'https://en.wikipedia.org/wiki/Theatre'
indepfilm_url = 'https://en.wikipedia.org/wiki/Independent_film'
fanfict_url = 'https://en.wikipedia.org/wiki/Fan_fiction'
indepanime_url = 'https://en.wikipedia.org/wiki/Independent_animation'
visualarts_url = 'https://en.wikipedia.org/wiki/Visual_arts'
workOart_url = 'https://en.wikipedia.org/wiki/Work_of_art'
filmmaking_url = 'https://en.wikipedia.org/wiki/Filmmaking'
play_url = 'https://en.wikipedia.org/wiki/Play_(theatre)'
musictheater_url = 'https://en.wikipedia.org/wiki/Musical_theatre'
anime_url = 'https://en.wikipedia.org/wiki/Animation'
fringe_url = 'https://en.wikipedia.org/wiki/Fringe_theatre'
offbroadway_url = 'https://en.wikipedia.org/wiki/Off-Broadway'
revue_url = 'https://en.wikipedia.org/wiki/Revue'
art_url = 'https://en.wikipedia.org/wiki/Art'
thearts_url = 'https://en.wikipedia.org/wiki/The_arts'
poetry_url = 'https://en.wikipedia.org/wiki/Poetry'
fineart_url = 'https://en.wikipedia.org/wiki/Fine_art'
cinejapan_url = 'https://en.wikipedia.org/wiki/Cinema_of_Japan'
magic_url = 'https://en.wikipedia.org/wiki/Magic_(illusion)'
entertainment_url = 'https://en.wikipedia.org/wiki/Entertainment'
performanceart_url = 'https://en.wikipedia.org/wiki/Performance_art'
cinepakistan_url = 'https://en.wikipedia.org/wiki/Cinema_of_Pakistan'


film_urls = [film_url, bolly_url, cineindia_url, holly_url, cineus_url, filmhist_url, movietheater_url, performarts_url, blockbuster_url, theatre_url, indepfilm_url, fanfict_url, indepanime_url, visualarts_url, workOart_url, filmmaking_url, play_url, musictheater_url, anime_url, fringe_url, offbroadway_url, revue_url, art_url, thearts_url, poetry_url, fineart_url, cinejapan_url, magic_url]

#Tech/Science URLs
tech_url = 'https://en.wikipedia.org/wiki/Technology'
techroad_url = 'https://en.wikipedia.org/wiki/Technology_roadmap'
techsociety_url = 'https://en.wikipedia.org/wiki/Technology_and_society'
techanalysis_url = 'https://en.wikipedia.org/wiki/Technical_analysis'
science_url = 'https://en.wikipedia.org/wiki/Science'
techwarI_url = 'https://en.wikipedia.org/wiki/Technology_during_World_War_I'
techwarII_url = 'https://en.wikipedia.org/wiki/Technology_during_World_War_II'
techconv_url = 'https://en.wikipedia.org/wiki/Technological_convergence'
techit_url = 'https://en.wikipedia.org/wiki/Information_technology'
techcomp_url = 'https://en.wikipedia.org/wiki/Computer'
techcompport_url='https://en.wikipedia.org/wiki/Portable_computer'
engineering_url = 'https://en.wikipedia.org/wiki/Engineering'
appscience_url = 'https://en.wikipedia.org/wiki/Applied_science'
appphysic_url = 'https://en.wikipedia.org/wiki/Applied_physics'
scientificmethod_url = 'https://en.wikipedia.org/wiki/Scientific_method'
scientist_url= 'https://en.wikipedia.org/wiki/Scientist'
sciencejournal_url = 'https://en.wikipedia.org/wiki/Scientific_journal'
emergtech_url = 'https://en.wikipedia.org/wiki/Emerging_technologies'
forcasting_url = 'https://en.wikipedia.org/wiki/Forecasting'
manufacture_url = 'https://en.wikipedia.org/wiki/Manufacturing'
industry_url = 'https://en.wikipedia.org/wiki/Industry'
logistics_url = 'https://en.wikipedia.org/wiki/Logistics'
industrev_url = 'https://en.wikipedia.org/wiki/Industrial_Revolution'
machinetool_url = 'https://en.wikipedia.org/wiki/Machine_tool'
carpentry_url = 'https://en.wikipedia.org/wiki/Carpentry'
geology_url = 'https://en.wikipedia.org/wiki/Geology'
            #History
techushist_url = 'https://en.wikipedia.org/wiki/Technological_and_industrial_history_of_the_United_States'
            #Computer related
machine_url = 'https://en.wikipedia.org/wiki/Machine'
telephone_url = 'https://en.wikipedia.org/wiki/Telephone'
mobilephone_url = 'https://en.wikipedia.org/wiki/Mobile_phone'
internet_url = 'https://en.wikipedia.org/wiki/Internet'
compnetwork_url = 'https://en.wikipedia.org/wiki/Computer_network'
radiofreq_url = 'https://en.wikipedia.org/wiki/Radio_frequency'
electronics_url = 'https://en.wikipedia.org/wiki/Electronics'
transmitter_url = 'https://en.wikipedia.org/wiki/Transmitter'
digitalsignal_url = 'https://en.wikipedia.org/wiki/Digital_signal'
digsubline_url = 'https://en.wikipedia.org/wiki/Digital_subscriber_line'
compscience_url = 'https://en.wikipedia.org/wiki/Computer_science'
telecomm_url = 'https://en.wikipedia.org/wiki/Telecommunication'
deploymentenvrn_url = 'https://en.wikipedia.org/wiki/Deployment_environment'
computer_url = 'https://en.wikipedia.org/wiki/Computer'
compgram_url = 'https://en.wikipedia.org/wiki/Computer_program'
compgramming_url = 'https://en.wikipedia.org/wiki/Computer_programming'
reverseengineer_url = 'https://en.wikipedia.org/wiki/Reverse_engineering'
sourcecode_url = 'https://en.wikipedia.org/wiki/Source_code'
dataprocess_url = 'https://en.wikipedia.org/wiki/Data_processing'
compsoftengine_url = 'https://en.wikipedia.org/wiki/Component-based_software_engineering#Software_component'
interchangpart_url = 'https://en.wikipedia.org/wiki/Interchangeable_parts'
            #Automobile related
            
            #Mathematical/Physical
algorithm_url = 'https://en.wikipedia.org/wiki/Algorithm'
electrification_url = 'https://en.wikipedia.org/wiki/Electrification'
            #Philosophy/Logic
            
            #Biology/Anatomy/Chemistry/Ecology&Geneology
primaryprod_url = 'https://en.wikipedia.org/wiki/Primary_production'
photosynth_url = 'https://en.wikipedia.org/wiki/Photosynthesis'
ecosyst_url = 'https://en.wikipedia.org/wiki/Ecosystem'
vegan_url = 'https://en.wikipedia.org/wiki/Herbivore'
protozoa_url = 'https://en.wikipedia.org/wiki/Protozoa'
chemsub_url = 'https://en.wikipedia.org/wiki/Chemical_substance'

tech_urls = [tech_url, techroad_url, techsociety_url, techushist_url, techanalysis_url, science_url, techwarI_url, techwarII_url, techconv_url, techit_url, techcomp_url, techcompport_url, engineering_url, appscience_url, appphysic_url, scientificmethod_url, scientist_url, sciencejournal_url, machine_url, telephone_url, mobilephone_url, internet_url, compnetwork_url, radiofreq_url, electronics_url, transmitter_url, digitalsignal_url, digsubline_url, compscience_url, telecomm_url, emergtech_url, forcasting_url, manufacture_url, industry_url, deploymentenvrn_url, logistics_url, industrev_url]

#Health URLs
health_url = 'https://en.wikipedia.org/wiki/Health'
healthus_url = 'https://en.wikipedia.org/wiki/Health_care_in_the_United_States'
healthins_url = 'https://en.wikipedia.org/wiki/Health_insurance'
healthcare_url = 'https://en.wikipedia.org/wiki/Health_care'
healthcan_url = 'https://en.wikipedia.org/wiki/Healthcare_in_Canada'
healthport_url = 'https://en.wikipedia.org/wiki/Health_Insurance_Portability_and_Accountability_Act'
healthtob_url = 'https://en.wikipedia.org/wiki/Health_effects_of_tobacco'

health_urls = [health_url, healthus_url, healthins_url, healthcare_url, healthcan_url,healthport_url, healthtob_url ]

#Cricket URLs (Included for reference to original implementation)
cric_url = 'https://en.wikipedia.org/wiki/Cricket'
t20_url = 'https://en.wikipedia.org/wiki/Twenty20'
test_url = 'https://en.wikipedia.org/wiki/Test_cricket'
wc_url = 'https://en.wikipedia.org/wiki/Cricket_World_Cup'
ipl_url = 'https://en.wikipedia.org/wiki/Indian_Premier_League'
t20wc_url = 'https://en.wikipedia.org/wiki/ICC_T20_World_Cup'
cricindia_url = 'https://en.wikipedia.org/wiki/Cricket_in_India'
indiacricteam_url = 'https://en.wikipedia.org/wiki/India_national_cricket_team'
histworldcup_url = 'https://en.wikipedia.org/wiki/History_of_the_ICC_Cricket_World_Cup'
champs_url = 'https://en.wikipedia.org/wiki/Champions_League_Twenty20'
engcric_url = 'https://en.wikipedia.org/wiki/Cricket#English_cricket_in_the_18th_and_19th_centuries'
ltdoverscric_url = 'https://en.wikipedia.org/wiki/Limited_overs_cricket'

cricket_urls = [cric_url,  t20_url, test_url, wc_url, ipl_url, t20wc_url, cricindia_url, indiacricteam_url, champs_url,
               engcric_url,ltdoverscric_url]

#Places urls

yakhini_url = 'https://en.wikipedia.org/wiki/Yakhini'
okukumapark_url = 'https://en.wikipedia.org/wiki/Okukuma_Prefectural_Natural_Park'
cucharasriver_url = 'https://en.wikipedia.org/wiki/Cucharas_River'
thejunction_url = 'https://en.wikipedia.org/wiki/The_Junction'
albuquerque_url = 'https://en.wikipedia.org/wiki/Old_Town_Albuquerque'
madinsaleh_url = 'https://en.wikipedia.org/wiki/Mada%27in_Saleh'
khirokitia_url = 'https://en.wikipedia.org/wiki/Khirokitia'
mountnemrut_url = 'https://en.wikipedia.org/wiki/Mount_Nemrut'
palaceofardashir_url = 'https://en.wikipedia.org/wiki/Palace_of_Ardashir'
nesvizhcastle_url = 'https://en.wikipedia.org/wiki/Nesvizh_Castle'
aggtelekslovakcaves_url = 'https://en.wikipedia.org/wiki/Caves_of_Aggtelek_Karst_and_Slovak_Karst'
krakowoldtown_url = 'https://en.wikipedia.org/wiki/Kraków_Old_Town'
novodevichyconvent_url = 'https://en.wikipedia.org/wiki/Novodevichy_Convent'
kazamkremlin_url = 'https://en.wikipedia.org/wiki/Kazan_Kremlin'
brunabionne_url = 'https://en.wikipedia.org/wiki/Brú_na_Bóinne'
saintsavinsurgartempe_url = 'https://en.wikipedia.org/wiki/Abbey_Church_of_Saint-Savin-sur-Gartempe'
wallsedwardgwynedd_url = 'https://en.wikipedia.org/wiki/Castles_and_Town_Walls_of_King_Edward_in_Gwynedd'
durhamcastle_url = 'https://en.wikipedia.org/wiki/Durham_Castle'
jeitaitemple_url = 'https://en.wikipedia.org/wiki/Jietai_Temple'
bellrocklighthouse_url = 'https://en.wikipedia.org/wiki/Bell_Rock_Lighthouse'



place_urls = [yakhini_url, okukumapark_url, cucharasriver_url, thejunction_url, albuquerque_url, madinsaleh_url, khirokitia_url, mountnemrut_url, palaceofardashir_url, nesvizhcastle_url, aggtelekslovakcaves_url, krakowoldtown_url, novodevichyconvent_url, kazamkremlin_url, brunabionne_url, saintsavinsurgartempe_url, wallsedwardgwynedd_url, durhamcastle_url, jeitaitemple_url, bellrocklighthouse_url]

#People urls

larshorntveth_url = 'https://en.wikipedia.org/wiki/Lars_Horntveth'
walterparsons_url = 'https://en.wikipedia.org/wiki/Walter_Parsons_(politician)'
hardycross_url = 'https://en.wikipedia.org/wiki/Hardy_Cross'
waynegibson_url = 'https://en.wikipedia.org/wiki/Wayne_Gibson'
ilghazi_url = 'https://en.wikipedia.org/wiki/Ilghazi'
francisbaring_url = 'https://en.wikipedia.org/wiki/Francis_Baring,_5th_Baron_Ashburton'
gayleharrell_url = 'https://en.wikipedia.org/wiki/Francis_Baring,_5th_Baron_Ashburton'
jonahanguka_url = 'https://en.wikipedia.org/wiki/Jonah_Anguka'


        #Celebrities and actors
waseemabbas_url = 'https://en.wikipedia.org/wiki/Waseem_Abbas'
tariqmustafa_url = 'https://en.wikipedia.org/wiki/Tariq_Mustafa'
DanishNawaz_url = 'https://en.wikipedia.org/wiki/Danish_Nawaz'

        #Inventors and scientists

        
        #social figures
jeanrousseau_url = 'https://en.wikipedia.org/wiki/Jean-Jacques_Rousseau'
georgeshaw_url = 'https://en.wikipedia.org/wiki/George_Bernard_Shaw'
robertounger_url = 'https://en.wikipedia.org/wiki/Roberto_Mangabeira_Unger'
shankarniyogi_url = 'https://en.wikipedia.org/wiki/Shankar_Guha_Niyogi'
ivanillich_url = 'https://en.wikipedia.org/wiki/Ivan_Illich'
jacobburckhardt_url = 'https://en.wikipedia.org/wiki/Jacob_Burckhardt'



people_urls = [larshorntveth_url, walterparsons_url, hardycross_url, waynegibson_url, waseemabbas_url, tariqmustafa_url, DanishNawaz_url, ilghazi_url, francisbaring_url, gayleharrell_url, jonahanguka_url, jeanrousseau_url, georgeshaw_url, robertounger_url, shankarniyogi_url, ivanillich_url, jacobburckhardt_url]

#initialize dataframe for urls
"""set topics for classifying and urls"""
all_categ_urls = [film_urls, tech_urls, health_urls, cricket_urls]
topics = ["Film", "Tech", "Health", "Cricket"]
iTopic = 0
df = pd.DataFrame(columns = ['Paragraph','Category'])
X = []
y = []

print("Preparing dataframe out of urls paragraph data...")
topic_limit = 2000  #adjust for capping data amounts
i = 0
para_count=0
page_indices=[]
y_index=[]
page_indices+=[para_count]
for categ_url in all_categ_urls:
    topic_ctr = 0
    limit_reached = 0
    topic = topics[iTopic]
    iTopic += 1
    #para_count=0
    for url in categ_url:
            if (limit_reached == 1):
                break
            r = requests.get(url)
            soup = BeautifulSoup(r.content, 'html.parser')
            table = soup.find_all('p',attrs={'class': None}) #finds paragraphs
            full_text = ''
            #para_count=0
            
  #iterate through urls and clean text while reading in
  
            for x in table:
                if(len(str(x.get_text().strip())) > 0):
                    topic_ctr = topic_ctr + 1
                    text = cleanText(str(x.get_text()))
                    df.loc[i] = [text,topic]  #word_tokenize(cleanText(str(x.get_text())))
                    y+=[topic]
                    i+=1
                    para_count+=1 
                    if (topic_ctr >= topic_limit ):
                        limit_reached = 1
                        break
            
            y_index+=[topic]
            page_indices+=[para_count]  #count paragraphs to keep track of pages
#            print("completed url: ", url, "\n %s as topic" %topic)
print("Succesfully created dataframe out of paragraphes with size: ", len(df))
print("\nNumer of articles processed is: ", (len(page_indices)-1))
print("\nListing top few rows w/ frequency distribution: ")

#create bag-of-words or frequency distribution to vectorize
#first data frame, then pages, then paragraphes
X_token=[]

from collections import Counter

df['Freq_Dist'] = df.Paragraph.apply(lambda x: Counter(x.split(" ")))

print(df.head(), "\n\n")

#Frequency distribution for whole pages
X_pages=[]
y_pages=[]

for i in range((len(page_indices)-1)):
    X_pages+=[sum(df.Freq_Dist[page_indices[i]:(page_indices[i+1]-1)], collections.Counter())]
    y_pages+=[y_index[i]]


#Frequency distribution for paragraphs

X_Freq_Dist=[]
y_labels=[]

for i in range(para_count):
    X_Freq_Dist+=[df.Freq_Dist[i]]
    y_labels+=[df.Category[i]]
    #if i%1000==0:print(type(df.Freq_Dist[i]))

#join paragraphs themselves

X_pag=[]
y_pag=[]
succ_ratio_list2=[]

for i in range(len(page_indices)-1):
    X_pag+=[(" ".join(df.Paragraph[page_indices[i]:(page_indices[i+1]-1)]))]
    y_pag+=[y_index[i]]

"""
# ngram count 2 words at a time for frequency distribution of pairings

cv = CountVectorizer(analyzer='word', ngram_range=(2, 2), max_df=0.5, max_features=10000, min_df=2, stop_words='english')
cv_trans = TfidfTransformer(norm='l2')

#for pag in X_pag:
cv_fit=cv.fit_transform(X_pag)
word_list = cv.get_feature_names();
count_list = cv_fit.toarray().sum(axis=0)
pairCounter_pag = dict(zip(word_list,count_list))
"""

#Use frequency distributions to call k-nn classifier made on homework 
#with different methods of measuring distance (takes forever and doesn't really work for classifying)
"""
succ_ratio_list1=[]
for j in range(30):
    X_train, X_test, y_train, y_test = train_test_split(X_pages, y_pages, test_size = 0.30)

    k=5
    trials1=[]
    sum_ones, sum_zeros = 0, 0
    for i in range(len(X_test)):
        predict1 = NN_countclassifier(X_train, X_test[i], k, y_test[i])
        classify_true = y_test[i]
    
        if predict1 == classify_true:
            trials1+=[1]
            sum_ones+=1
        elif predict1!=classify_true:
            trials1+=[0]
            sum_zeros+=1
    succ_ratio_list1+=[(sum_ones/len(X_test))]
    #print((sum_ones/len(X_test)))
print("mean for 30 trials: ", np.mean(succ_ratio_list1))
#print("successful labellings out of total with %d NN counter classifier is:\n" % k, (sum_ones/len(X_test)))
"""


"""
for j in range(30):

    X_train_, X_test_, y_train_, y_test_ = train_test_split(X_pag, y_pag, test_size = 0.30)

    k=5
    trials2=[]
    sum_ones, sum_zeros = 0, 0
    for i in range(len(X_test_)):
        predict2 = NN_wordsclassifier(X_train_, X_test_[i], k, y_test_[i])
        classify_true = y_test_[i]
    
        if predict2 == classify_true:
            trials2+=[1]
            sum_ones+=1
        elif predict2!=classify_true:
            trials2+=[0]
            sum_zeros+=1
    succ_ratio_list2+=[(sum_ones/len(X_test))]
print("mean for 30 trials: ", np.mean(succ_ratio_list2))
#print("successful labellings out of total with %d NN phrases classifier is:\n" % k, (sum_ones/len(X_test)))

"""

#regather paragraphs for sklearn's K-nearest neighbors classifier test
#use CountVectorizer to get paragraphs in correct format
#Frequency Distribution of entire pages
#(term frequency - inverse document frequency)
#               tf-idf

# Vectorize
bow_vectorizer = CountVectorizer(max_df=0.5, max_features=10000, min_df=2, stop_words='english')

X = bow_vectorizer.fit_transform(X_pag)

from sklearn.neighbors import KNeighborsClassifier

# Classification algorithm testing accuracy for 30 trials
knn = KNeighborsClassifier(n_neighbors = 12, algorithm='brute', metric='cosine')

dtm = pd.DataFrame(X.toarray(), columns=bow_vectorizer.get_feature_names())   # X or cv_fit.toarray    col=bow_vectorizer.get_feature_names() or word_list
#print(dtm.head())
succ_ratio_list3=[]
for i in range(30):
    X_Train, X_Test, y_Train, y_Test = train_test_split(dtm, y_pag, test_size=0.20)

    knn.fit(X_Train, y_Train)
    predict = knn.predict(X_Test)      #bowTest[0:5000])  X[:2]
    j=0
    trials3=[]
    sum_ones, sum_zeros = 0, 0
    for word in predict:
        if word==y_Test[j]:
                trials3+=[1]
                sum_ones+=1
        elif word!=y_Test[j]:
                trials3+=[0]
                sum_zeros+=1
    succ_ratio_list3+=[(sum_ones/len(X_Test))]
    #print((sum_ones/len(X_test)))
    j+=1
print("  Actual number of last trial's countvectorizer features: ", X_Train.shape[1])
print("Mean for 30 trials using rudimentary Frequency Distribution: ", np.mean(succ_ratio_list3))

#print("Predicted categories of wiki-pages:\n", predict, "\nThe true categories:\n", y_Test)

#Logistic regression
logisticModel = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
logisticModel.fit(X_Train,y_Train)
print("Logistic regression: ", logisticModel.score(X_Test,y_Test))
print("\nMean cross value score (offset of accuracy measure): ", cross_val_score(logisticModel, dtm, y_pag,cv=4).mean())


#get dtm columns to be used as training and test
columnList = list(dtm.columns.values)
trainingColumns = int(len(columnList)-1)
X = columnList[:trainingColumns]
y = columnList[:len(columnList)-trainingColumns]

#RandomForestRegressor   might not work with only few categories needs adjusted
from sklearn.ensemble import RandomForestRegressor
X_train1, X_test1, y_train1, y_test1 = train_test_split(dtm,dtm[y],test_size=0.30)
randomModel = RandomForestRegressor(n_estimators=100, random_state=42)
randomModel.fit(X_train1, y_train1)
print("Random forest regressor: ", randomModel.score(X_test1,y_test1))


#Semantics LSA vector classification

import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier

#Vectorizer specified below, then applies SVD to project onto N
#principle components that would be a loose semantic representation
#of the articles then classify with knn classifier specified
#to cosine similarity or brute force algrotihmic approach
########################################################################
#  Use LSA to vectorize the articles.
########################################################################

for n in range(1, 4):
  for j in range(30):

    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_pag,y_pag,test_size=0.30)

# Tfidf vectorizer:
#   - Strips out “stop words”
#   - Filters out terms that occur in more than half of the docs (max_df=0.5)
#   - Filters out terms that occur in only one document (min_df=2).
#   - Selects the 10,000 most frequently occuring words in the corpus.
#   - Normalizes the vector (L2 norm of 1.0) to normalize the effect of
#     document length on the tf-idf values.
    vectorizer = TfidfVectorizer(max_df=0.55, max_features=10000, min_df=2, stop_words='english', norm=None, use_idf=False)

# Build the tfidf vectorizer from the training data and apply it
    X_train_tfidf = vectorizer.fit_transform(X_train2)
    
    #print("  Actual number of tfidf features: %d" % X_train_tfidf.get_shape()[1])

    #print("\nPerforming dimensionality reduction using LSA")
    #t0 = time.time()

# Project the tfidf vectors onto the first 100 principal components.
# Though this is significantly fewer features than the original tfidf vector,
# they are stronger features, and the accuracy is higher.
    svd = TruncatedSVD(100)
    lsa = make_pipeline(svd, Normalizer(norm='l2', copy=False))

# Run SVD on the training data, then project the training data.
    X_train_lsa = lsa.fit_transform(X_train_tfidf)

    #print("  done in %.3fsec" % (time.time() - t0))

    explained_variance = svd.explained_variance_ratio_.sum()
    #print("  Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))


# Now apply the transformations to the test data as well.
    X_test_tfidf = vectorizer.transform(X_test2)
    X_test_lsa = lsa.transform(X_test_tfidf)



########################################################################
#  Run classification of the test articles
########################################################################

    #print("\nClassifying tfidf vectors...")

# Time this step.
#t0 = time.time()

# Build a k-NN classifier. Use k = 5 (majority wins), the cosine distance,
# and brute-force calculation of distances.


    knn_tfidf = KNeighborsClassifier(n_neighbors=n, algorithm='brute', metric='cosine')
    knn_tfidf.fit(X_train_tfidf, y_train2) #X_train_tfidf, y_train2

# Classify the test vectors.
    p = knn_tfidf.predict(X_test_tfidf)

# Measure accuracy
    accur_tflist = []
    numRight = 0;
    for i in range(0,len(p)):
        if p[i] == y_test2[i]:
            numRight += 1
    accur_tflist+=[(float(numRight) / float(len(y_test2)) * 100.0)]
    #print("  (%d / %d) correct - %.2f%%" % (numRight, len(y_test2), float(numRight) / float(len(y_test2)) * 100.0))

# Calculate the elapsed time (in seconds)
    #elapsed = (time.time() - t0)
    #print("  done in %.3fsec" % elapsed)

    #print("\nClassifying LSA vectors...")

    # Time this step.
    #t0 = time.time()

# Build a k-NN classifier. Use k = 5 (majority wins), the cosine distance,
# and brute-force calculation of distances.
    knn_lsa = KNeighborsClassifier(n_neighbors=n, algorithm='brute', metric='cosine')
    knn_lsa.fit(X_train_lsa, y_train2)

# Classify the test vectors.
    p = knn_lsa.predict(X_test_lsa)

# Measure accuracy
    accur_list=[]
    numRight = 0;
    for i in range(0,len(p)):
        if p[i] == y_test2[i]:
            numRight += 1
    accur_list+=[(float(numRight) / float(len(y_test2)) * 100.0)]
#print("  (%d / %d) correct - %.2f%%" % (numRight, len(y_test2), float(numRight) / float(len(y_test2)) * 100.0))

  print("\nMean accuracy for tf-idf vector %d classification: " % n, np.mean(accur_tflist), "\n\nMean accuracy for LSA vector %d classification: " % n, np.mean(accur_list))

# Calculate the elapsed time (in seconds)
#elapsed = (time.time() - t0)
#print("    done in %.3fsec" % elapsed)


from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans

#run kmeans on training data from LSA/SVD calculations
#k = len(topics)
k_list=[]
accurate_list=[]
for k in range(len(topics)-1, len(topics)+6):
  km = KMeans(n_clusters=k, init='k-means++', max_iter=150, n_init=10) #n_init=1
  if k==len(topics)-1:print("Clustering sparse data with %s" % km)
  metrics1=[]
  metrics2=[]
  metrics3=[]
  metrics4=[]
  metrics5=[]

  t0 = time.time()
  for i in range(30):
    km.fit(X_train_lsa)
#    labels_list=[]
#    for elt in y_train2:
#        if elt=="Film": labels_list+=[0]
#        elif elt=="Tech": labels_list+=[1]
        
    metrics1+=[metrics.homogeneity_score(y_train2, km.labels_)]
    metrics2+=[metrics.completeness_score(y_train2, km.labels_)]
    metrics3+=[metrics.v_measure_score(y_train2, km.labels_)]
    metrics4+=[metrics.adjusted_rand_score(y_train2, km.labels_)]
    metrics5+=[metrics.silhouette_score(X_train_lsa, km.labels_, sample_size=1000)]
  print("%d clusters assumed" % k)
  k_list+=[k]
  print("Homogeneity mean: %0.3f" % np.mean(metrics1))
  accurate_list+=[np.mean(metrics1)]
  print("Completeness mean: %0.3f" % np.mean(metrics2))
  print("V-measure mean: %0.3f" % np.mean(metrics3))
  print("Adjusted Rand-Index mean: %.3f"
      % np.mean(metrics4))
  print("Silhouette Coefficient mean: %0.3f"
      % np.mean(metrics5))
print("\nActual number of clusters is ", len(topics))

import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,5))
plt.plot(k_list, accurate_list, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('number of clusters')
plt.ylabel('Homogeneity of labels vs k-mean prediction')
leg = plt.legend(['Accuracy for K-means Clustering'], loc='best', borderpad=0.3,
                 shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                 markerscale=0.4)
leg.get_frame().set_alpha(0.4)
#leg.draggable(state=True)
#plt.text(4, .7, topics, fontsize=10, verticalalignment='bottom', horizontalalignment='right')
plt.show()
