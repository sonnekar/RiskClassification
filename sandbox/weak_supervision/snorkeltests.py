from snorkel.labeling import LabelingFunction, PandasLFApplier
from textblob import TextBlob
import pandas as pd
import numpy as np

# Define weak labeling functions with expanded short synonyms
def lf_keyword_frequency(report):
    report = report['report'] 
    keywords = [
        "injury", "accident", "incident", "harm", "damage", "trauma", "wound", "casualty", 
        "fatality", "mishap", "disaster", "loss", "break", "shock", "collapse", "trauma", 
        "hit", "fall", "safety", "risk", "crash", "spill", "burn", "cut", "abrasion", "injured"
    ]
    count = sum(1 for word in report.lower().split() if word in keywords)
    if count > 2:  
        return 1  
    elif count == 1: 
        return 2  
    return 0  

def lf_energy_indicators(report):
    report = report['report']  
    high_energy_indicators = [
        "shock", "explosion", "fire", "voltage", "gravity", "pressure", "fall", "impact", 
        "electrocution", "burn", "radiation", "sudden", "hazard", "chemical", "safety", 
        "risk", "kinetic", "arc", "force", "crane", "heavy", "machine", "load"
    ]
    low_energy_indicators = [
        "slip", "trip", "minor", "safe", "light", "routine", "normal", "non-life", 
        "manual", "soft", "limited", "low-risk", "minor", "quick", "superficial"
    ]
    
    if any(word in report.lower() for word in high_energy_indicators):
        return 1  
    elif any(word in report.lower() for word in low_energy_indicators):
        return 2  
    return 0  

def lf_negation(report):
    report = report['report']
    negations = [
        "not injured", "no accidents", "no injuries", "none", "without incident", "did not occur", 
        "no harm", "no effect", "no damage", "didn't happen", "nothing occurred", "zero incidents"
    ]
    if any(negation in report.lower() for negation in negations):
        return 0  
    return -1  

def lf_risk_assessment(report):
    report = report['report']
    risk_keywords = [
        "risk", "danger", "hazard", "threat", "exposure", "unsafe", "alert", 
        "warning", "potential", "issue", "problem", "concern"
    ]
    if any(keyword in report.lower() for keyword in risk_keywords):
        return 3  
    return -1  

def lf_injury_severity(report):
    report = report['report']
    serious_keywords = [
        "serious", "critical", "severe", "grave", "life", "major", "fatal", 
        "trauma", "catastrophic", "extreme", "permanent", "debilitating"
    ]
    minor_keywords = [
        "minor", "slight", "mild", "temporary", "light", "soft", "non-critical", 
        "shallow", "reversible", "superficial"
    ]
    
    if any(keyword in report.lower() for keyword in serious_keywords):
        return 1  
    elif any(keyword in report.lower() for keyword in minor_keywords):
        return 2  
    return -1  

def lf_energy_context(report):
    report = report['report']
    high_energy_contexts = [
        "voltage", "power", "transformer", "generator", "pressure", "heavy", 
        "elevation", "danger", "risk", "hazard", "explosive"
    ]
    low_energy_contexts = [
        "manual", "routine", "low", "safe", "hand", "light", "normal"
    ]
    
    if any(context in report.lower() for context in high_energy_contexts):
        return 1  
    elif any(context in report.lower() for context in low_energy_contexts):
        return 2  
    return -1  

def lf_temporal_context(report):
    report = report['report']
    temporal_keywords = [
        "recent", "now", "earlier", "last", "before", "just", "previous"
    ]
    if any(keyword in report.lower() for keyword in temporal_keywords):
        return 3  
    return -1  

def lf_adjective_presence(report):
    report = report['report']
    risk_adjectives = [
        "unsafe", "dangerous", "hazardous", "risky", "volatile", "threatening", 
        "urgent", "imminent", "serious", "grave", "extreme"
    ]
    if any(adj in report.lower() for adj in risk_adjectives):
        return 3  
    return -1  

def lf_personnel_role(report):
    report = report['report']
    role_keywords = [
        "engineer", "worker", "supervisor", "technician", "manager", "operator", 
        "foreman", "safety", "staff", "crew", "electrician"
    ]
    if any(role in report.lower() for role in role_keywords):
        return -1  
    return -1  

def lf_sentiment_analysis(report):
    report = report['report']
    sentiment = TextBlob(report).sentiment.polarity
    if sentiment < 0:
        return 3  
    return -1  

def lf_action_words(report):
    report = report['report']
    action_words = [
        "fail", "drop", "collide", "crash", "misstep", "overload", 
        "spill", "misalign", "strike", "hurt", "damage", "lose"
    ]
    if any(action in report.lower() for action in action_words):
        return 3  
    return -1  

def lf_proximity_to_energy(report):
    report = report['report']
    proximity_keywords = [
        "near", "close", "adjacent", "within", "danger", "hazard", 
        "proximity", "striking", "range", "reach"
    ]
    if any(keyword in report.lower() for keyword in proximity_keywords):
        return 3  
    return -1  

def lf_reporting_style(report):
    report = report['report']
    if report.startswith("-"):  
        return -1  
    return -1  

def lf_safety_protocols(report):
    report = report['report']
    safety_protocols = [
        "safety", "protocol", "procedure", "measure", "check", "audit", 
        "regulation", "control", "guard", "barrier"
    ]
    if any(protocol in report.lower() for protocol in safety_protocols):
        return 0  
    return -1  

lfs = [
    LabelingFunction(name="lf_keyword_frequency", f=lf_keyword_frequency),
    LabelingFunction(name="lf_energy_indicators", f=lf_energy_indicators),
    LabelingFunction(name="lf_negation", f=lf_negation),
    LabelingFunction(name="lf_risk_assessment", f=lf_risk_assessment),
    LabelingFunction(name="lf_injury_severity", f=lf_injury_severity),
    LabelingFunction(name="lf_energy_context", f=lf_energy_context),
    LabelingFunction(name="lf_temporal_context", f=lf_temporal_context),
    LabelingFunction(name="lf_adjective_presence", f=lf_adjective_presence),
    LabelingFunction(name="lf_personnel_role", f=lf_personnel_role),
    LabelingFunction(name="lf_sentiment_analysis", f=lf_sentiment_analysis),
    LabelingFunction(name="lf_action_words", f=lf_action_words),
    LabelingFunction(name="lf_proximity_to_energy", f=lf_proximity_to_energy),
    LabelingFunction(name="lf_reporting_style", f=lf_reporting_style),
    LabelingFunction(name="lf_safety_protocols", f=lf_safety_protocols),
]

df = pd.read_csv('../dataset/data.csv')
df.fillna("NA", inplace=True)
df['DATETIME_DTM'] = pd.to_datetime(df['DATETIME_DTM'])
df.set_index('DATETIME_DTM', inplace=True)
df['report'] = df['PNT_NM'] + df['QUALIFIER_TXT'] + df['PNT_ATRISKNOTES_TX'] + df['PNT_ATRISKFOLWUPNTS_TX']
df = df[['report']]
print(df.head())

applier = PandasLFApplier(lfs=lfs)
Y = applier.apply(df=df[['report']]) 

label_mapping = {0: 'None', 1: 'HSIF', 2: 'LSIF', 3: 'PSIF'}

def majority_vote(row):
    valid_labels = row[row != -1] 
    if len(valid_labels) == 0:  
        return 0 
    return np.bincount(valid_labels).argmax()  

weak_labels = np.apply_along_axis(majority_vote, 1, Y)

df['weak_label'] = weak_labels

y = np.array(df['weak_label'], dtype=np.int32)
print((y==0).sum(),(y==1).sum(),(y==2).sum(),(y==3).sum())
