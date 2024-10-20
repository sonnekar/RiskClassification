

import ollama

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

df = pd.read_csv('../dataset/data.csv')




df.set_index('OBSRVTN_NB', inplace=True)



df.fillna("NA", inplace=True)



df['DATETIME_DTM'] = pd.to_datetime(df['DATETIME_DTM'])



df['PNT_NM'].nunique(), df['PNT_NM'].nunique() / len(df)


df_subsample = df.sample(10)



PROMPT = """
<|im_start|>system 
{0}
<|im_end|>

<|im_start|>user
{1}
<|im_end|>

<|im_start|>assistant
{2}
"""


class OneDecisionTest:
    def __init__(
            self,
            system_prompt: str,
            df: pd.DataFrame = df_subsample,
            model: str = 'llama3:latest',
            classes: Tuple[str] = ('True', 'False'),
        ):
        """Generate one decision for an llm to make. 

        Args:
            system_prompt (str): _description_
            classes (Tuple[str]): _description_
            df (pd.DataFrame): _description_
            model: _description
        """

        self._output = ollama.chat(
            model    = model, 
            messages = [{'role': 'system', 'content': system_prompt,},
                        {'role': 'user', 'content': 'what is 1+1?'}],#f'Based on the following info {df}, classify the data into these classes {classes}. Output only the class AND NOTHING ELSE.',}],
            #stream  = True,
            options  = {'temperature': 0}
        )
        print(self._output['message']['content'])
        # for chunk in self._llm_stream:
            # print(chunk['message']['content'], end='', flush=True)

OneDecisionTest(system_prompt='Your job is to do math.')


SYSTEM_PROMPT_FORMAT = """
You are a superintelligent Safety Classification and Learning (SCL) Model with the goal of classifying field notes written on-site by AEP electrical engineers and other essential personnel.

Your task is to reason with the field notes and output classifications for that field note data. If there are figures in the class definitions, but not figures in the input data, use your best judgment to infer which class it belongs to.

The field notes have the following schema:
- PNT_NM: Point name, the safety criteria being assessed. Forms a primary key when combined with OBSRVN_NB (observation number).
- QUALIFIER_TXT: Qualifier text, list of predetermined observations chosen by the reviewer based on the point being assessed.
- PNT_ATRISKNOTES_TX: Point at-risk notes text, comments left by observer regarding unsafe conditions they found.
- PNT_ATRISKFOLWUPNTS_TX: Point at-risk follow-up notes text, recommended remediation for at-risk conditions observed. This field may be empty (Indicated by "NA"). 

You must classify this data using the following criteria:

### **Classification for Serious Injury and Fatality (SIF) Risk:**

1. **Serious Injury Classification Criteria:**
   - **Fatalities**: Death resulting from the injury.
   - **Amputations**: Involving bone.
   - **Concussions and/or cerebral hemorrhages**: Include all cerebral hemorrhages and severe concussions resulting in loss of consciousness or symptoms lasting more than 24 hours.
   - **Injury or trauma to internal organs**: Serious if objective medical evidence indicates significant or sustained organ damage or progressive changes in organ function.
   - **Bone fractures**: Include open, compound, or comminuted fractures of fingers and toes. Include all bone fractures of the face, skull, or navicular wrist bone. Exclude hairline fractures, except those in the face, skull, or navicular wrist bone.
   - **Complete tears**: Includes complete tendon, ligament, and cartilage tears of major joints.
   - **Herniated disks**: Neck or back.
   - **Lacerations**: Resulting in severed tendons or deep wounds requiring internal stitches.
   - **Burns**: 2nd (10% body surface) or 3rd degree burns.
   - **Eye injuries**: Resulting in eye damage or loss of vision.
   - **Injections of foreign materials**: (e.g., hydraulic fluid).
   - **Severe heat exhaustion and heat stroke cases**.
   - **Dislocation of a major joint requiring manipulation by a doctor**.
   - **Other Injuries**: Select for reporting injuries not identified in the existing categories.

2. **Direct Control Evaluation:**
   Evaluate the field note to determine if a **Direct Control** is present. A Direct Control is defined as follows:
   - **Targeted**: The barrier must specifically target a high-energy source.
   - **Effectiveness**: It must effectively mitigate exposure to the high-energy source when installed, verified, and used properly (i.e., a SIF should not occur if these conditions are met).
   - **Unintentional Human Error**: It must be effective even in the presence of unintentional human error during work that is unrelated to the installation of the control.

   **Examples of Direct Controls:**
   - Lockout/Tagout (LOTO)
   - Machine guarding
   - Hard physical barriers
   - Fall protection
   - Cover-ups

   **Examples that are NOT Direct Controls:**
   - Training
   - Warning signs
   - Rules
   - Experience
   - Standard non-specialized personal protective equipment (e.g., hard hats, gloves, boots)

3. **High Energy Situation:**
High energy refers to situations where physical forces or hazardous conditions are strong enough to increase the likelihood of a serious injury or fatality (SIF). Examples include falls from elevation, suspended loads, rotating equipment, explosions, electrical contact over 50 volts, high temperatures, or high pressure.

4. **High Energy Incident:**
A high energy incident occurs when a high energy situation results in an actual event, injury, or near-miss, involving hazardous energy being released and the worker coming into contact with or in proximity to the energy source.



Based on the provided field notes, you must classify whether the reported incident qualifies as a **High Energy Situation**, **High Energy Incident**, **Serious Injury** and if a **Direct Control** is present.

NOTE: Your output should follow **ONLY this JSON schema**! (i.e., just output the classification and NOTHING else!)
schema:
{{
  "high_energy_situation_classification": "{hes_keys}",
  "high_energy_incident_classification": "{hei_keys}"
  "serious_injury_classification": "{injury_keys}",
  "direct_control_classification": "{control_keys}",
  
}}
"""



USER_PROMPT_FORMAT = """
Below are the field notes:

{field_notes}

REMEMBER: 
your output should follow ONLY this json schema! (i.e. just output the classification and NOTHING else!)
schema:
{{ 'high_energy_situation_classification': enum{hes_keys}, 'high_energy_incident_classification': enum{hei_keys}, 'serious_injury_classification': enum{injury_keys}, 'direct_control_classification': enum{control_keys} }}
"""



class OneDecision:
    def __init__(
            self,
            hes_classes: Dict[str, str],
            hei_classes: Dict[str, str],
            injury_classes: Dict[str, str],
            control_classes: Dict[str, str],
            column_label: str,
            df: pd.DataFrame = df_subsample,
            model: str = 'llama3:latest',
        ):  
        """Generate decisions for LLM to classify both Serious Injury and Direct Control."""

        prompt_hes_keys    = str(list(hes_classes.keys()))
        
        prompt_hei_keys    = str(list(hei_classes.keys()))
        prompt_injury_keys    = str(list(injury_classes.keys()))
        
        prompt_control_keys    = str(list(control_classes.keys()))

        for _, row in df.iterrows():
            prompt_field_notes = row[['PNT_NM', 'QUALIFIER_TXT', 'PNT_ATRISKNOTES_TX', 'PNT_ATRISKFOLWUPNTS_TX']].to_frame().to_string(index=True, header=False, max_colwidth=None)        
            
            print(prompt_field_notes)

            output = ollama.chat(
                model    = model, 
                messages = [{'role': 'system', 'content': SYSTEM_PROMPT_FORMAT.format(hes_keys = prompt_hes_keys, hei_keys = prompt_hei_keys, injury_keys=prompt_injury_keys, control_keys=prompt_control_keys)},
                            {'role': 'user', 'content': USER_PROMPT_FORMAT.format(field_notes=prompt_field_notes, hes_keys = prompt_hes_keys, hei_keys = prompt_hei_keys, injury_keys=prompt_injury_keys, control_keys=prompt_control_keys)}],
                options  = {'temperature': 0}
            )

            print(output['message']['content'])

        # df[column_label] = predictions


# incident_prompts = {
#     "suspended load": "A suspended load exceeding 500 lbs lifted more than 1 foot off the ground typically requires special equipment. This scenario exceeds the high-energy threshold.",
#     "fall from elevation": "A fall from over 4 feet of elevation, considering an average human weight of 150 lbs, exceeds the high-energy threshold.",
#     "Mobile Equipment/Traffic with Workers on Foot": "Most mobile equipment or vehicles in motion exceed the high-energy threshold due to their mass. The risk is from the worker's perspective, not the operator’s. In work zones, this applies when a vehicle veers within 6 feet of a worker or if a worker enters the traffic pattern.",
#     "Motor Vehicle incident (occupant)": "A motor vehicle crash at speeds of 30 mph or more exceeds the high-energy threshold, focusing on vehicle occupants, including the driver.",
#     "Heavy rotating equipment": "Heavy rotating equipment, beyond powered hand tools, typically exceeds the high-energy threshold due to the complexity of calculating mechanical energy.",
#     "High Temperature": "Substances at or above 150°F can cause third-degree burns with 2 seconds of contact, surpassing the high-energy threshold.",
#     "Steam": "Steam release always exceeds the high-energy threshold due to its high temperature and potential for burns.",
#     "Fire with Sustained Fuel Source": "Fires fueled by materials like paper burn at around 700°F, well above the high-energy threshold.",
#     "Explosion": "Any explosion exceeds the high-energy threshold due to the immense force released.",
#     "Excavation fr Trench": "Trench or excavation exposure deeper than 5 feet exceeds the high-energy threshold. Soil pressure increases by 40 lbs/ft² per foot of depth, reaching 200 lbs/ft² at 5 feet.",
#     "Electrical Contact with source": "Electrical contact at 50 volts or higher can cause serious injury or death, exceeding the high-energy threshold, per NFPA 70E.",
#     "Arc Flash": "Any arc flash exceeds the high-energy threshold due to high voltage. OSHA Standard 1910.333 defines safe distances.",
#     "High Dose of Chemical or Radiation": "Exposure to toxic chemicals, radiation, or environments reducing oxygen below 16 percent surpasses the high-energy threshold. Corrosive chemicals with pH <2 or >12.5 should be assessed by qualified personnel using IDLH values from the CDC."  
# }

serious_injury_present = {
    'Serious Injury': 'Description of serious injury criteria',
    'No Serious Injury': 'Does not meet serious injury criteria'
}

direct_control_present = {
    'Direct Control Present': 'Effective control targeting high energy source is present',
    'No Direct Control': 'No direct control is present'
}

high_energy_situation = {
    'High Energy Situation': 'A situation where the presence of high energy increases the likelihood of serious injury or fatality.',
    'No High Energy Situation': 'No high energy conditions are present.'
}

high_energy_incident = {
    'High Energy Incident': 'A high energy situation that resulted in an actual incident, injury, or near-miss.',
    'No High Energy Incident': 'No high energy incident occurred.'
}

OneDecision(
    hes_classes = high_energy_situation,
    hei_classes = high_energy_incident,
    injury_classes = serious_injury_present, 
    control_classes = direct_control_present, 
    column_label = 'inv'
)



df_subsample





