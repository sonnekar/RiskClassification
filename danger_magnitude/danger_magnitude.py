import ollama

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from final_draft.snorkelfuncs import lf_keyword_frequency,lf_energy_indicators,lf_negation,lf_risk_assessment,lf_injury_severity,lf_energy_context,lf_temporal_context,lf_adjective_presence,lf_personnel_role,lf_sentiment_analysis,lf_action_words,lf_proximity_to_energy,lf_reporting_style,lf_safety_protocols
from snorkel.labeling import LabelingFunction, PandasLFApplier

from scipy.stats import zscore

SYSTEM_PROMPT = """
You are a superintelligent Safety Classification and Learning (SCL) Model 
with the goal of measuring the injury and danger levels from field notes written on site by AEP electrical engineers and other essential personnel.

The field notes have the following schema:
PNT_NM: Point name, the safety criteria being assessed. Forms a primary key when combined with OBSRVN_NB (observation number)
QUALIFIER_TXT: Qualifier text, list of predetermined observations chosen by the reviewer based on the point being assessed
PNT_ATRISKNOTES_TX: Point at-risk notes text, comments left by observer regarding unsafe conditions they found
PNT_ATRISKFOLWUPNTS_TX: Point at-risk follow up notes text, recommended remediation for at-risk conditions observed. This field may be empty (Indicated by "NA").

For your purpose, 'danger or injury' means people (AEP personell OR other), property 
(be it AEP, public, or personal), or the environment is at a risk of or actually did have inflicted harm / damage.

Your output should be a score from 1-10 and NOTHING else, with 10 being the highest injury and danger levels.
"""

USER_PROMPT = """
Here is the report:
```
{report}
```
Remember:
Your output should be a score from 1-10 and NOTHING else, with 10 being the highest injury and danger levels.
"""

MULTISHOT = {
    """
    PNT_NM: Climbing - Procedures
    QUALIFIER_TXT: Was a drop zone established, and clearly marked?
    PNT_ATRISKNOTES_TX: Workers were hoisting beams and steel onto a scaffold. A certified mechanic operated an air hoist to lift the beam. As the lift was performed, the rigging was caught under an adjacent beam. Under the increasing tension, the cable snapped and struck a second employee in the leg, fully fracturing his femur. An investigation indicated that the rigging was not properly inspected before the lift.
    PNT_ATRISKFOLWUPNTS_TX: NA
    """: "9",
    """
    PNT_NM: Housekeeping - Generation
    QUALIFIER_TXT: Job site hazards, Tripping Hazards
    PNT_ATRISKNOTES_TX: An employee was descending a staircase and when stepping down from the last step she rolled her
    ankle on an extension cord on the floor. She suffered a torn ligament and a broken ankle that resulted in
    persistent pain for more than a year.
    PNT_ATRISKFOLWUPNTS_TX: NA 
    """: "7",
    """
    PNT_NM: Workplace Conditions Addressed
    QUALIFIER_TXT: Voltage being worked discussed
    PNT_ATRISKNOTES_TX: A crew was working near a sedimentation pond on a rainy day. The boom of the trac-hoe was within 3
    feet of a live 12kV line running across the road. No contact was made because a worker intervened and
    communicated with the operator.
    PNT_ATRISKFOLWUPNTS_TX: NA
    """: "3",
    """
    PNT_NM: Did you recognize additional slip, trip, or fall hazards that had not already been recognized and mitigated? If so, please select or describe these hazards in the At-Risk notes.
    QUALIFIER_TXT: Awareness of environment
    PNT_ATRISKNOTES_TX: [NAME] was working a near by cliff that had about a 20' drop off, crew didn't discuss as a hazard on briefing, i discussed with GF and he told the foreman to make the corrections and place something out there to give crews a visual.
    PNT_ATRISKFOLWUPNTS_TX: NA
    """: "6",
    """
    PNT_NM: PPE
    QUALIFIER_TXT: Side Shields adequate
    PNT_ATRISKNOTES_TX: Emplolyee was witnessed without side shields. Supervosor was informed so he couldd coach his employee.,
    PNT_ATRISKFOLWUPNTS_TX: NA
    """: "2"
}

class GenerateDangerMagnitudes:
    def __init__(
            self,
            df: pd.DataFrame,
            shot_examples: Optional[Dict[str, str]] = None,
            model: str = 'llama3:latest',
        ):  
        """Generate danger magnitude score / metric with LLM (model) using shot examples from df.
        
        Adds it to df under 'danger_magnitude'. """
        predictions = []

        for _, row in tqdm(df.iterrows()):
            prompt_field_notes = "\n".join(f"{key}: {row[key]}" for key in ['PNT_NM', 'QUALIFIER_TXT', 'PNT_ATRISKNOTES_TX', 'PNT_ATRISKFOLWUPNTS_TX'])
            #print(prompt_field_notes)

            output = ollama.chat(
                model    = model, 
                messages = self._build_shot_prompts(prompt_field_notes, shot_examples),
                options  = {'temperature': 0, 'num_predict': 2}
            )

            #print(output['message']['content'])
            predictions.append(self._validate_output(output['message']['content']))

        df['danger_magnitude'] = predictions
        
    def _build_shot_prompts(self, prompt_field_notes, shot_examples):
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]

        if shot_examples is None:
            messages.append({'role': 'user', 'content': USER_PROMPT.format(report=prompt_field_notes)})
            #self._pretty_print_messages(messages)
            return messages

        for example in shot_examples:
            messages.append({'role': 'user', 'content': USER_PROMPT.format(report=example)})
            messages.append({'role': 'assistant', 'content': str(shot_examples[example])})

        messages.append({'role': 'user', 'content': USER_PROMPT.format(report=example)})
        #self._pretty_print_messages(messages)
        return messages
    
    def _validate_output(self, output: str) -> int:
        if output.isnumeric():
            if 0 < int(output) < 10:
                return int(output)
        return 0

    def _pretty_print_messages(self, messages):
        for message in messages:
            role = message['role'].capitalize()
            content = message['content']
            print(f"{role}: {content}\n")

class GenerateWeakLabels:
    def __init__(self, df: pd.DatetimeIndex):
        """Generate weak labels for HSIF, LSIF, PSIF, None with Snorkel.
        
        Adds it to df under 'weak_label'. """
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

        df_local = df.copy()
        df_local['report'] = df_local['PNT_NM'] + df_local['QUALIFIER_TXT'] + df_local['PNT_ATRISKNOTES_TX'] + df_local['PNT_ATRISKFOLWUPNTS_TX']
        df_local = df_local[['report']]

        applier = PandasLFApplier(lfs=lfs)
        Y = applier.apply(df=df_local) 

        def majority_vote(row):
            valid_labels = row[row != -1] 
            if len(valid_labels) == 0:  
                return 0 
            return np.bincount(valid_labels).argmax()  

        df['weak_label'] = np.apply_along_axis(majority_vote, 1, Y)

if __name__ == '__main__':
    df = pd.read_csv('../dataset/data.csv')

    df.set_index('OBSRVTN_NB', inplace=True)
    df.fillna("NA", inplace=True)
    df['DATETIME_DTM'] = pd.to_datetime(df['DATETIME_DTM'])

    GenerateWeakLabels(df=df)
    #GenerateDangerMagnitudes(df=df, shot_examples=MULTISHOT)

    df['ovr_danger'] = zscore(df['weak_label'])# + (zscore(df['danger_magnitude']) * 2)

    df.to_pickle('df.pkl')
