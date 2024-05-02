from bigDictionariesLexicons import * 
from bigDictionariesAdjectives import *
from bigDictionariesInquirer import *
from collections import OrderedDict
from polesConstructs import *
from testData import *

LexiconsEnsembl = [harvardGeneralInquirer3623,WEAT1, vaderLexicon7062,NRCEmotionLexicon5555,opinionLexicon6786,afinnLexicon2477,
        positiveNegativeAdjetives762,positiveNegativeAdjetives197,happySadAdjectives122,niceMeanAdjetives228,intelligentDullAdjetives75,
        inquirerViceVirtue1277,inquirerHostileAffiliation1176,inquirerPowerConflictCooperation294,inquirerAffectNegativePositive261,
        ideonomyPersonalityTraits526, EMOTEvalence985,EMOTELikeableness985, EMOTELikeableness554    
        ]

dataDict={


    '0-1': {
        'name': 'bipolar  Occupations',
        'constructPole1' : maleConstruct,
        'constructPole2' : femaleConstruct,
        'RealDataLexicons' : [percentageOccupationFemale]
    },
    
    '0-2': {
        'name': 'bipolar  GDP per capita',
        'constructPole1' : poorCountriesConstruct,
        'constructPole2' : richCountriesConstruct,
        'RealDataLexicons' : [countriesToGdpDictionary],
    },    
    
    '0-3': {
        'name': 'bipolar Car Brand prices',
        'constructPole1' : cheapCarsConstruct,
        'constructPole2' : expensiveCarsConstruct,
        'RealDataLexicons' : [carPrices],
    },    

  
    '0-4': {
        'name': 'bipolar  US Senators and party affiliation',
        'constructPole1' : republicansConstruct,
        'constructPole2' : democratsConstruct,        
        'RealDataLexicons' : [senatorsUS],
    }, 
    
    '0-6': {
        'name': 'Death & Life',
        'constructPole1' : deathConstruct,
        'constructPole2' : lifeConstruct,
        'RealDataLexicons' : LexiconsEnsembl,
        'axisName' : "Death to Life Axis",
    },		
        
    
    '0-7': {
        'name': 'Disease & Health',
        'constructPole1' : diseaseConstruct,
        'constructPole2' : healthConstruct,
        'RealDataLexicons' : LexiconsEnsembl,
        'axisName' : "Disease to Health Axis",
    },

    '0-8': {
        'name': 'Dictatorship & Democracy',
        'constructPole1' : dictatorshipConstruct,
        'constructPole2' : democracyConstruct,
        'RealDataLexicons' : LexiconsEnsembl,
        'axisName' : "Dictatorship to Democracy Axis",
    },    
	
    '0-9': {
        'name': 'Malevolent & Respectable Figures',
        'constructPole1' : evilPeopleConstruct,
        'constructPole2' : goodPeopleConstruct,
        'RealDataLexicons' : LexiconsEnsembl,
        'axisName' : "Malevolent to Respectable Axis",
    },  
    

    '1-1': {
        'name': 'personal ideology',
        'constructPole1' : conservativesConstruct,
        'constructPole2' : liberalsConstruct,
        'RealDataLexicons' : LexiconsEnsembl,
        'axisName' : "Conservatives to Liberals Axis",                  
    },     

    '1-2': {
        'name': 'party affiliation and political parties',
        'constructPole1' : republicansConstruct,
        'constructPole2' : democratsConstruct,
        'RealDataLexicons' : LexiconsEnsembl,
        'axisName' : "Republicans to Democrats Axis",                  
    },    

    '1-3': {
        'name': 'U.S. presidents',
        'constructPole1' : republicanPresidents,
        'constructPole2' : democratPresidents,
        'RealDataLexicons' : LexiconsEnsembl,
        'axisName' : "RepublicanPresidents to DemocratPresidents Axis",                  
    },	

    '1-4': {
        'name': 'Ideologically Oriented Journalists',
        'constructPole1' : rightwingJournalists,
        'constructPole2' : leftwingJournalists,
        'RealDataLexicons' : LexiconsEnsembl,
        'axisName' : "RightwingJournalists to LeftwingJournalists Axis",  
    },    

    '1-5': {
        'name': 'U.S. senators',
        'constructPole1' : republicanSenators,
        'constructPole2' : democratSenators,
        'RealDataLexicons' : LexiconsEnsembl,
        'axisName' : "RepublicanSenators to DemocratSenators Axis",  
    },     

    '1-6': {
        'name': 'influential conservatives and liberals',
        'constructPole1' : influentialConservativesConstruct,
        'constructPole2' : influentialLiberalsConstruct,
        'RealDataLexicons' : LexiconsEnsembl,
        'axisName' : "InfluentialConservatives to InfluentialLiberals Axis",  
    },	

    
}

