from textblob import TextBlob

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
        return 3  
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
        return 3  
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
        return 1  
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
        return 3  
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
        return 3  
    elif any(context in report.lower() for context in low_energy_contexts):
        return 2  
    return -1  

def lf_temporal_context(report):
    report = report['report']
    temporal_keywords = [
        "recent", "now", "earlier", "last", "before", "just", "previous"
    ]
    if any(keyword in report.lower() for keyword in temporal_keywords):
        return 1  
    return -1  

def lf_adjective_presence(report):
    report = report['report']
    risk_adjectives = [
        "unsafe", "dangerous", "hazardous", "risky", "volatile", "threatening", 
        "urgent", "imminent", "serious", "grave", "extreme"
    ]
    if any(adj in report.lower() for adj in risk_adjectives):
        return 1  
    return -1  

def lf_personnel_role(report):
    report = report['report']
    role_keywords = [
        "engineer", "worker", "supervisor", "technician", "manager", "operator", 
        "foreman", "safety", "staff", "crew", "electrician"
    ]
    if any(role in report.lower() for role in role_keywords):
        return 0  
    return -1  

def lf_sentiment_analysis(report):
    report = report['report']
    sentiment = TextBlob(report).sentiment.polarity
    if sentiment < 0:
        return 1  
    return -1  

def lf_action_words(report):
    report = report['report']
    action_words = [
        "fail", "drop", "collide", "crash", "misstep", "overload", 
        "spill", "misalign", "strike", "hurt", "damage", "lose"
    ]
    if any(action in report.lower() for action in action_words):
        return 1  
    return -1  

def lf_proximity_to_energy(report):
    report = report['report']
    proximity_keywords = [
        "near", "close", "adjacent", "within", "danger", "hazard", 
        "proximity", "striking", "range", "reach"
    ]
    if any(keyword in report.lower() for keyword in proximity_keywords):
        return 1  
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