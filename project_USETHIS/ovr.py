import ollama

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from snorkelfuncs import lf_keyword_frequency,lf_energy_indicators,lf_negation,lf_risk_assessment,lf_injury_severity,lf_energy_context,lf_temporal_context,lf_adjective_presence,lf_personnel_role,lf_sentiment_analysis,lf_action_words,lf_proximity_to_energy,lf_reporting_style,lf_safety_protocols
from snorkel.labeling import LabelingFunction, PandasLFApplier

from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
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
    """: "4",
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
    """: "3",
    """
    PNT_NM: PPE
    QUALIFIER_TXT: Other - PPE, Safety glasses adequate
    PNT_ATRISKNOTES_TX: Employee operating Digger [NAME] not wearing safety glasses. Two not wearing correct safety glasses. One wearing safety glasses without side shields. One employee marking and drilling pole without work gloves.
    PNT_ATRISKFOLWUPNTS_TX: NA
    """: "5",
    """
    PNT_NM: 15) If new hazards were identified, or if conditions changed since the original briefing, were they documented?
    QUALIFIER_TXT: [Notes Required for At-Risk Conditions]
    PNT_ATRISKNOTES_TX: The crew was using a cart / gator too move scaffolding material later in the day and one of the guy was not wearing a seat belt, I stopped him and explain to him that he needed to wear the seatbelt at all times.
    PNT_ATRISKFOLWUPNTS_TX: NA
    """: "7",
}

ASCENDING_SEVERITY_EXAMPLES = {
    0: "no incidents no hazards safe environment routine operations "
                "no unsafe conditions no accidents all clear safe practices "
                "standard procedure followed no reported issues positive observation "
                "fully compliant safe workplace no anomalies zero risk",
    
    1: "slip trip basic safety hazard minor injury standard protocol follow-up "
                "routine checks low energy situation precautionary measures "
                "safety reminder light load normal operations general safety "
                "minimal risk administrative controls slight concern safe practices "
                "non-critical observation", 

    2: "slip trip fall overhead load heavy machinery PPE unsafe conditions "
                   "moderate injury caution hazard near miss temporary risk "
                   "potential danger minor accident equipment malfunction "
                   "vulnerable position safety protocol observed routine risk assessment "
                   "increased caution recommended",

    3: "fire explosion electrical fall hazardous energy heavy equipment "
                 "high voltage pressure shock electrocution crane collision trauma "
                 "critical failure severe hazard unsafe conditions risk exposure "
                 "catastrophic incident dangerous situation urgent action required "
                 "immediate danger serious injury potential fatality severe consequences",
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

        messages.append({'role': 'user', 'content': USER_PROMPT.format(report=prompt_field_notes)})
        #self._pretty_print_messages(messages)
        return messages
    
    def _validate_output(self, output: str) -> int:
        if output.isnumeric():
            if 0 < int(output) <= 10:
                return int(output)
        return 0

    def _pretty_print_messages(self, messages):
        for message in messages:
            role = message['role'].capitalize()
            content = message['content']
            print(f"{role}: {content}\n")

class GenerateWeakLabels:
    def __init__(self, df: pd.DataFrame):
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


class GenerateNaiveCosDistanceClusteringLabels:
    def __init__(
            self,
            df: pd.DataFrame,
            examples: Dict[int, str] = ASCENDING_SEVERITY_EXAMPLES
        ):
        """
        Estimate danger of hazards by taking the cosine distance of an 
        incoming report embedding against a set of increasing severity examples.
        """
        self.df = df
        self.examples = examples
        
        # Generate embeddings for the severity examples
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Example model
        self.example_embeddings = self._embed_examples()
        
        # Assign severity labels to the DataFrame
        self._assign_severity_labels()

    def _embed_examples(self):
        example_texts = list(self.examples.values())
        # Move to CPU and convert to NumPy array
        return self.model.encode(example_texts, convert_to_tensor=True).cpu().numpy()

    def _assign_severity_labels(self):
        self.df['report'] = self.df['PNT_NM'] + self.df['QUALIFIER_TXT'] + self.df['PNT_ATRISKNOTES_TX'] + self.df['PNT_ATRISKFOLWUPNTS_TX']
        # Generate embeddings for the DataFrame reports
        report_embeddings = self.model.encode(self.df['report'].tolist(), convert_to_tensor=True).cpu().numpy()

        # Calculate cosine similarity between report embeddings and example embeddings
        similarities = cosine_similarity(report_embeddings, self.example_embeddings)

        # Assign the key of the most similar example to a new column
        self.df['cos_cluster_label'] = np.argmax(similarities, axis=1)

class GenerateSimpleBinaryClassification:
    def __init__(self, df: pd.DataFrame, col_name: str, class_def: str, model: str = 'llama3:latest'):
        """Prompt llm to enter a 1 where a row falls under class_def, 0 otherwise.
        
        The classification corresponding with 0 should mean an estimated lower danger."""
        ex = """
            Determine whether the Input Data should be classified as this definition:
            If a High-Energy Criteria is present, output a 1. High Energy Criteria: Suspended Loads: Loads greater than 500 lbs lifted more than 1 foot off the ground. Elevation: Any height exceeding 4 feet, considering the average weight of a human (over 150 lbs). Mobile Equipment: Most mobile equipment, including motor vehicles, exceeds the high-energy threshold when in motion from the perspective of a worker on foot. Work Zone Traffic: An incident occurs if a vehicle departs its intended path within 6 feet of an exposed employee or if an employee enters the traffic pattern. Motor Vehicle Speed: An estimate of 30 miles per hour is considered the high-energy threshold for serious or fatal crashes. Mechanical Energy: Heavy rotating equipment beyond powered hand tools typically exceeds the high-energy threshold. Temperature: Exposure to substances at or above 150 degrees Fahrenheit can cause third-degree burns if contacted for 2 seconds or more. Any release of steam or combustion materials (e.g., paper burning at approximately 700 degrees Fahrenheit) exceeds the high-energy threshold. Explosions: Any incident described as an explosion exceeds the high-energy threshold. Unsupported Soil: Soil in a trench or excavation exceeding 5 feet of height, with pressure increasing approximately 40 pounds per square foot for each foot of depth. Electrical Energy: Voltage equal to or exceeding 50 volts can result in serious injury or death. Any arc flash also exceeds the high-energy threshold. Toxic Chemicals or Radiation: Use IDLH (Immediately Dangerous to Life or Health) values from the CDC and consider: Oxygen (O2) levels below 16% Corrosive chemical exposures (pH <2 or >12.5) Additional Considerations: If a situation does not fit any of the above criteria but encompasses potential energy: Estimate the weight in pounds (lbs) and height in feet (ft). If height is greater than 503.1 × weight − 0.99 503.1×weight −0.99 , it can be classified as high energy. If it encompasses kinetic energy: Estimate the weight in pounds (lbs) and speed in miles per hour (mph). If speed is greater than 𝑦 = 182.71 × weight − 0.68 , it can be classified as high energy.
            
            Input Data:
            ```
            'was working a near by cliff that had about a 20\' drop off'
            ```

            Your output should be a 1 if you classify the Input Data under the definition,
            and a 0 if you do NOT classify the Input Data under the definition.

            Output ONLY a 0 or 1 and NOTHING more
        """

        predictions = []

        for _, row in tqdm(df.iterrows()):
            prompt_field_notes = "\n".join(f"{key}: {row[key]}" for key in ['PNT_NM', 'QUALIFIER_TXT', 'PNT_ATRISKNOTES_TX', 'PNT_ATRISKFOLWUPNTS_TX'])
            prompt = f"""
                Determine whether the Input Data should be classified as this definition:
                {class_def}

                Input Data:
                ```
                {prompt_field_notes}
                ```

                Your output should be a 1 if you classify the Input Data under the definition,
                and a 0 otherwise.

                Output ONLY a 0 or 1 and NOTHING more
                """
            output = ollama.chat(
                model=model,
                messages=[{'role': 'user', 'content': ex}, {'role': 'user', 'content': '1'}, {'role': 'user', 'content': prompt}],
                options  = {'temperature': 0, 'num_predict': 1}
            )
            cleaned_out = self._validate_output(output['message']['content'])
            #print(cleaned_out)
            predictions.append(cleaned_out)

        df[col_name] = predictions

    def _validate_output(self, output: str) -> int:
        if output.isnumeric():
            if 0 <= int(output) <= 1:
                return int(output)
        #print('here', output)
        return 0


if __name__ == '__main__':
    df = pd.read_csv('../dataset/data.csv')

    df.set_index('OBSRVTN_NB', inplace=True)
    df.fillna("NA", inplace=True)
    df['DATETIME_DTM'] = pd.to_datetime(df['DATETIME_DTM'])

    df = df.sample(n=1000, random_state=42)

    GenerateWeakLabels(df=df)
    GenerateNaiveCosDistanceClusteringLabels(df=df)
    GenerateDangerMagnitudes(df=df, shot_examples=MULTISHOT)
    GenerateSimpleBinaryClassification(df=df, col_name='HIS', class_def="If a High-Energy Criteria is present, output a 1. High Energy Criteria: Suspended Loads: Loads greater than 500 lbs lifted more than 1 foot off the ground. Elevation: Any height exceeding 4 feet, considering the average weight of a human (over 150 lbs). Mobile Equipment: Most mobile equipment, including motor vehicles, exceeds the high-energy threshold when in motion from the perspective of a worker on foot. Work Zone Traffic: An incident occurs if a vehicle departs its intended path within 6 feet of an exposed employee or if an employee enters the traffic pattern. Motor Vehicle Speed: An estimate of 30 miles per hour is considered the high-energy threshold for serious or fatal crashes. Mechanical Energy: Heavy rotating equipment beyond powered hand tools typically exceeds the high-energy threshold. Temperature: Exposure to substances at or above 150 degrees Fahrenheit can cause third-degree burns if contacted for 2 seconds or more. Any release of steam or combustion materials (e.g., paper burning at approximately 700 degrees Fahrenheit) exceeds the high-energy threshold. Explosions: Any incident described as an explosion exceeds the high-energy threshold. Unsupported Soil: Soil in a trench or excavation exceeding 5 feet of height, with pressure increasing approximately 40 pounds per square foot for each foot of depth. Electrical Energy: Voltage equal to or exceeding 50 volts can result in serious injury or death. Any arc flash also exceeds the high-energy threshold. Toxic Chemicals or Radiation: Use IDLH (Immediately Dangerous to Life or Health) values from the CDC and consider: Oxygen (O2) levels below 16% Corrosive chemical exposures (pH <2 or >12.5) Additional Considerations: If a situation does not fit any of the above criteria but encompasses potential energy: Estimate the weight in pounds (lbs) and height in feet (ft). If height is greater than 503.1 × weight − 0.99 503.1×weight −0.99 , it can be classified as high energy. If it encompasses kinetic energy: Estimate the weight in pounds (lbs) and speed in miles per hour (mph). If speed is greater than 𝑦 = 182.71 × weight − 0.68 , it can be classified as high energy.")
    GenerateSimpleBinaryClassification(df=df, col_name='HII', class_def="If a High-Energy Incident is present, output a 1. High-Energy Incident: A high-energy incident is defined as an occurrence where: A high-energy source is released, changing state and becoming hazardous in the work environment The energy source must be no longer contained or under the control of the worker. The worker must have either: Contact: direct transmission of high energy to the human body. Proximity: within 6 feet of the energy source with unrestricted egress, or any distance in confined spaces or situations where escape is restricted.")
    GenerateSimpleBinaryClassification(df=df, col_name='IS', class_def="If there is a severe injury sustained in the data, output a 1.")
    GenerateSimpleBinaryClassification(df=df, col_name='NDC', class_def="If there is NO Mention of installed safety measures such as Machine guarding, Hard physical barriers, Fall protection systems, Covers or shield, output a 1")

    df['ovr_danger'] = (df['weak_label'] / 4) + (df['cos_cluster_label'] / 4) + (df['danger_magnitude'] / 5) + (df['HII'] / 4) + (df['IS'] / 4) + (df['NDC'] / 4) + (df['HIS'] / 4)

    scaler = MinMaxScaler()
    df['ovr_danger'] = scaler.fit_transform(df[['ovr_danger']])

    df.to_pickle('df2.pkl')
