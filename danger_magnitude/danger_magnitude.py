import ollama

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


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

Your output should be a score from 0-10 and NOTHING else, with 10 being the highest injury and danger levels.
"""

USER_PROMPT = """
Here is the report:
```
{report}
```
Remember:
Your output should be a score from 0-10 and NOTHING else, with 10 being the highest injury and danger levels.
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
            shot_examples: Optional[Dict[str, str]] = None,
            model: str = 'llama3:latest',
        ):  
        """Generate one decision for an llm to make. """
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

if __name__ == '__main__':
    df = pd.read_csv('../dataset/data.csv')

    df.set_index('OBSRVTN_NB', inplace=True)
    df.fillna("NA", inplace=True)
    df['DATETIME_DTM'] = pd.to_datetime(df['DATETIME_DTM'])

    GenerateDangerMagnitudes(shot_examples=MULTISHOT)