from bigDictionariesLexicons import * 
from bigDictionariesAdjectives import *
from bigDictionariesInquirer import *
from collections import OrderedDict

professions = ['elementary_teacher', 'registered_nurse', 'secretary', 'administrative_assistant', 'nurse', 'customer_service', 'manager', 'retail_sales', 'cashier', 'administrative_support', 'accountant', 'auditor', 'receptionist', 'maid', 'housekeeping', 'financial_manager', 'teaching_assistant', 'waitress', 'social_worker', 'preschool_teacher', 'kindergarten_teacher', 'cook', 'truck_driver', 'driver', 'laborer', 'freighter', 'construction_worker', 'janitor', 'software_developer', 'sales_representative', 'maintenance_worker', 'carpenter', 'chief_executive', 'mechanic', 'stock_clerk', 'electrician', 'general_manager', 'operations_manager', 'sales_manager', 'marketing_manager', 'police_officer']

politicalIssues = ['agriculture', 'civil_rights', 'commerce', 'defense', 'education', 'energy', 'environment', 'foreign_affairs', 'government_operations', 'health', 'housing', 'immigration', 'labor', 'law_and_crime', 'macroeconomics', 'public_lands', 'science_and_communications', 'social_welfare', 'trade','transportation']

deathConstruct = ['death','dying','decease']
lifeConstruct = ['alive','life','living']

maleConstruct = ['man','men','male','males']
femaleConstruct = ['woman','women','female','females']

poorCountriesConstruct = ['poor','poverty','underdeveloped']
richCountriesConstruct = ['wealth','rich','wealthy','prosperous','developed']

cheapCarsConstruct = ['affordable','budget','cheap','low_cost','poor','bargain','economical','inexpensive']
expensiveCarsConstruct =  ['expensive','rich','prosperous','wealthy','affluent','luxurious','wealth','lavish','upscale','pricey']

diseaseConstruct = ['disease','sick','sickness','illness']
healthConstruct = ['health','healthy','well_being']

dictatorshipConstruct = ['dictatorship','dictator','dictators', 'autocracy','authoritarianism','totalitarianism','tyranny','despotism']
democracyConstruct = ['democracy','democratic_leader','democratic_leaders','representative_government']

#https://www.thetoptens.com/most-evil-people-in-history/
evilPeopleConstruct = ['Hitler','Stalin','Bin_Laden','Pol_Pot','Heinrich_Himmler','Saddam_Hussein','Joseph_Goebbels'] #
goodPeopleConstruct = ['Gandhi','MLK','Nelson_Mandela','Mother_Teresa','Abraham_Lincoln']


conservativesConstruct = ['conservative','conservatives','right_winger','rightwinger','right_wingers','rightwingers','right_wing','rightwing','right_leaning']
liberalsConstruct = ['liberal','liberals','progressive','progressives','left_winger','leftwinger','left_wingers','leftwingers','left_wing','leftwing','left_leaning']

republicansConstruct = ['Republican', 'Republicans','GOP','Republican_Party',]
democratsConstruct = ['Democrat', 'Democrats','Democratic_Party']

republicanPresidents = ['Dwight_Eisenhower','Richard_Nixon','Gerald_Ford','Ronald_Reagan','George_Bush','Donald_Trump'] 
democratPresidents = ['Franklin_Roosevelt','Harry_Truman','John_Kennedy','Lyndon_Johnson','Jimmy_Carter','Bill_Clinton','Barack_Obama']

#https://www.politico.com/blogs/media/2015/04/twitters-most-influential-political-journalists-205510
rightwingJournalists = ['Jake_Tapper','Megyn_Kelly','Sean_Hannity','Michelle_Malkin','Dana_Perino','Bret_Baier','Greta_Van Susteren','Glenn_Beck','Bill_Reilly','Andrew_Malcolm','Matt_Drudge','Charles_Krauthammer','Ann_Coulter','Ed_Henry','Dana_Loesch','Brit_Hume','Sarah_Elizabeth_Cupp','Major_Garrett','Greg_Gutfeld','Tucker_Carlson','Andrea_Tantaros','Andrew_Napolitano','Erick_Erickson','Stephen_Hayes','Kimberly_Guilfoyl','Jonah_Goldberg','Neil_Cavuto','Peggy_Noonan','Monica_Crowley','Kirsten_Powers','Robert_Costa','Larry_Sabato','Mary_Katharine_Ham','Eric_Bolling','Rich_Lowry']
leftwingJournalists=['Anderson_Cooper','Rachel_Maddow','Ezra_Klein','Arianna_Huffington','Nate_Silver','George_Stephanopoulos','Christiane_Amanpour','Paul_Krugman','Ann_Curry','Chris_Hayes','Glenn_Greenwald','Melissa_Harris_Perry','Fareed_Zakaria','Donna_Brazile','Nicholas_Kristof','John_Dickerson','David_Corn','Robert_Reich','Katrina_vanden_Heuvel','Jim_Roberts','Matt_Taibbi','Matthew_Yglesias','Lawrence_Donnell','Andy_Borowitz','Chris_Matthews','Diane_Sawyer','Don_Lemon','Markos_Moulitsas','Thomas_Friedman','Ana_Marie_Cox','Chris_Cuomo','Al_Sharpton','Andrew_Sullivan','Bill_Keller','Charles_Blow',]

#List of US senators 28/2/2020 https://en.wikipedia.org/wiki/List_of_current_United_States_senators
republicanSenators = ['Mike_Pence','Richard_Shelby','Dan_Sullivan','Lisa_Murkowski','Martha_McSally','Tom_Cotton',
'John_Boozman','Cory_Gardner','Rick_Scott','Marco_Rubio','David_Perdue','Kelly_Loeffler','Jim_Risch',
'Mike_Crapo','Todd_Young','Mik_ Braun','Chuck_Grassley','Joni_Ernst','Pat_Roberts','Jerry_Moran','Rand_Paul',
'Mitch_McConnell','John_Kennedy','Bill_Cassidy','Susan_Collins','Roger_Wicker','Cindy_Hyde_Smith','Josh_Hawley','Roy_Blunt',
'Steve_Daines','Ben_Sasse','Deb_Fischer','Thom_Tillis','Richard_Burr','John_Hoeven','Kevin_Cramer','Rob_Portman','James_Lankford','Jim_Inhofe','Pat_Toomey','Tim_Scott','Lindsey_Graham','John_Thune','Mike_Rounds','Marsha_Blackburn','Lamar_Alexander','Ted_Cruz','John_Cornyn','Mitt_Romney','Mike_Lee','Shelley_Moore_Capito','Ron_Johnson','Mike_Enzi','John_Barrasso']

democratSenators = ['Doug_Jones','Kyrsten_Sinema','Kamala_Harris','Dianne_Feinstein','Michael_Bennet','Chris_Murphy',
'Richard_Blumenthal','Chris_Coons','Tom_Carper','Brian_Schatz','Mazie_Hirono','Dick_Durbin','Tammy_Duckworth',
'Chris_Van_Hollen','Ben_Cardin','Elizabeth_Warren','Ed_Markey','Debbie_Stabenow','Gary_Peters','Tina_Smith','Amy_Klobuchar',
'Jon_Tester','Jacky_Rosen','Catherine_Cortez_Masto','Jeanne_Shaheen','Maggie_Hassan','Bob_Menendez',
'Cory_Booker','Tom_Udall','Martin_Heinrich','Chuck_Schumer','Kirsten_Gillibrand','Sherrod_Brown','Ron_Wyden',
'Jeff_Merkley','Bob_Casey','Sheldon_Whitehouse','Jack_Reed','Patrick_Leahy','Mark_Warner','Tim_Kaine','Patty_Murray','Maria_Cantwell','Joe_Manchin','Tammy_Baldwin','Bernie_Sanders',
]


#https://www.telegraph.co.uk/news/worldnews/northamerica/usa/6990965/The-most-influential-US-conservatives-20-1.html
# excluding names already included in previous constructs
influentialConservativesConstruct =  ['Dick_Cheney','Rush_Limbaugh','Sarah_Palin','Robert_Gates',
                      'Roger_Ailes','David_Petraeus','Paul_Ryan','Tim_Pawlenty','John_Roberts','Haley_Barbour','Eric_Cantor','John_McCain','Bob_McDonnell','Newt_Gingrich','Mike_Huckabee',
                      ]
					 
influentialLiberalsConstruct = ['Hillary_Clinton','Nancy_Pelosi','Rahm_Emanuel','Al_Gore','Oprah_Winfrey',
                     'Tim_Geithner','David_Axelrod','Harry_Reid','Michelle_Obama','Arianna_Huffington',
                     'Sonia_Sotomayor','Denis_McDonough','Janet_Napolitano','Mark_Warner','Robert_Gibbs',
                      'Barney_Frank','John_Kerry','Eric_Holder',]



# 1. 支持堕胎权 vs 反对堕胎权
# proChoiceConstruct = ['pro_choice', 'reproductive_rights', 'bodily_autonomy', 'abortion_access', 'Planned_Parenthood']
                      
# proLifeConstruct = ['pro_life', 'right_to_life', 'protect_unborn', 'abortion_is_murder', 'life_begins_at_conception']
                    

proChoiceConstruct =  ['Cecile_Richards', 'Leana_Wen', 'Ilyse_Hogue', 'Kimberly_Inez_McGuire', 'Renee_Bracey_Sherman',  # Pro-Choice activists
                       'NARAL_Pro_Choice_America', 'Planned_Parenthood_Action_Fund', 'Center_for_Reproductive_Rights', 'National_Organization_for_Women', 'EMILY_List',  # Pro-Choice organizations
                       'Tammy_Baldwin', 'Kirsten_Gillibrand', 'Kamala_Harris', 'Mazie_Hirono', 'Patty_Murray'  # Pro-Choice politicians
                       ]
proLifeConstruct = ['Lila_Rose', 'Abby_Johnson', 'Kristan_Hawkins', 'Marjorie_Dannenfelser', 'Carol_Tobias',  # Pro-Life activists
                    'National_Right_to_Life_Committee', 'Americans_United_for_Life', 'Susan_B_Anthony_List', 'American_Life_League', 'Family_Research_Council',  # Pro-Life organizations'Dan_Lipinski', 
                    'Brian_Fitzpatrick', 'John_Katko', 'Collin_Peterson', 'Ben_McAdams'  # Pro-Life politicians
                    ]

# proChoiceConstruct = [
#     'Planned_Parenthood', 'NARAL_Pro-Choice_America', 'Center_for_Reproductive_Rights', 
#     'Reproductive_Health_Access_Project', 'Guttmacher_Institute', "Whole_Woman's_Health",
#     'National_Organization_for_Women', "EMILY's_List", 'A_Is_For', 'National_Abortion_Federation',
#     'Roe_v_Wade', 'Doe_v_Bolton'
# ]

# proLifeConstruct = [
#     'National_Right_to_Life_Committee', 'Americans_United_for_Life', 'Susan_B_Anthony_List',
#     'American_Life_League', 'Priests_for_Life', 'March_for_Life_Action', 'Live_Action', 
#     'Students_for_Life_of_America', 'Family_Research_Council', 'Concerned_Women_for_America',
#     'Roe_v_Wade', 'Gonzales_v_Carhart'
    
# ]




# 2. 支持health care改革 vs 反对health care改革 
# prohealthCareReformConstruct = ['support_health_care_reform', 'affordable_care_act', 'Obamacare', 'expand_Medicaid', 'universal_health_coverage', 'single_payer_healthcare', 'Medicare_for_all']
# antihealthCareReformConstruct = ['oppose_health_care_reform', 'socialized_medicine', 'government_takeover_of_healthcare', 'repeal_Obamacare', 'keep_private_insurance']

prohealthCareReformConstruct = ['affordable_care', 'expand_coverage', 'protect_preexisting_conditions', 'lower_premiums', 'Medicaid_expansion']
antihealthCareReformConstruct = ['government_overreach', 'higher_taxes', 'rationing_care', 'lose_current_plan', 'doctor_patient_relationship']

# 3. 宏观经济表现良好 vs 宏观经济表现不好
# goodMacroEconomyConstruct = ['strong_economy', 'low_unemployment', 'GDP_growth', 'bull_market', 'economic_boom', 'rising_wages', 'strong_consumer_confidence']
# badMacroEconomyConstruct = ['recession', 'high_unemployment', 'economic_downturn', 
# 'bear_market', 'market_crash', 'sluggish_growth', 'stagnant_wages']

goodMacroEconomyConstruct = ['strong_growth', 'low_unemployment', 'booming_stock_market', 'rising_wages', 'consumer_confidence']
badMacroEconomyConstruct =  ['recession', 'high_unemployment', 'market_crash', 'stagnant_wages', 'economic_anxiety']

# 4. 移民对美国有益 vs 移民对美国有害
proImmigrationConstruct = [
    'diversity', 'DACA', 'DREAMER', 'H1B', 'STEM', 'asylum', 'refugee', 'path_to_citizenship'
]

antiImmigrationConstruct = [
     'undocumented', 'deportation', 'enforcement', 'ICE', 'anchor_baby', 'border_wall',
    'border_wall', 'E-Verify', 'sanctuary_cities', 'amnesty', 'chain_migration'
]

# proImmigrationConstruct = ['diversity_is_strength', 'immigrant_rights', 'refugee_welcome', 'path_to_citizenship', 'immigration_reform']

# antiImmigrationConstruct = ['secure_borders', 'illegal_aliens', 'deport_criminals', 'immigration_enforcement', 'protect_jobs']