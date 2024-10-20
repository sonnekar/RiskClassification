import ollama
import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('data.csv')
df = df.sample(n=10, random_state=42)  # Set random_state for reproducibility


SYSTEM_PROMPT = """
Based on the following patient notes, please answer the questions with 'yes' or 'no' only.

High Energy Criteria:
Does the incident meet any of the following criteria?

Suspended Loads: Loads over 500 lbs lifted more than 1 foot.
Elevation: Heights exceeding 4 feet.
Mobile Equipment: Mobile equipment exceeds the high-energy threshold when in motion.
Work Zone Traffic: A vehicle departs its path within 6 feet of an exposed employee.
Motor Vehicle Speed: Estimated speed of 30 mph or greater.
Mechanical Energy: Heavy rotating equipment beyond powered hand tools.
Temperature: Contact with substances at or above 150°F.
Explosions: Any incident described as an explosion.
Unsupported Soil: Soil in a trench exceeding 5 feet in height.
Electrical Energy: Voltage equal to or exceeding 50 volts.
Toxic Chemicals or Radiation: IDLH values from the CDC, including oxygen levels below 16% or corrosive chemical exposures (pH <2 or >12.5).
If any of these criteria apply, respond with yes; otherwise, respond with no.

Criteria Answer: [Answer]

High-Energy Incident:
A high-energy incident is defined as an occurrence where:

A high-energy source is released, changing state and becoming hazardous in the work environment.
The energy source must be no longer contained or under the control of the worker.
The worker must have either:
Contact: direct transmission of high energy to the human body.
Proximity: within 6 feet of the energy source with unrestricted egress, or any distance in confined spaces or situations where escape is restricted.
Does the event described meet the criteria for a high-energy incident? Incident Answer: [Answer]

Severe Injury:
Is there a severe injury in the example? Severe Injury Answer: [Answer]

Direct Control:
A direct control is defined as one that:

Is specifically targeted at a high-energy source.
Effectively mitigates exposure to the high-energy source when installed, verified, and used properly (i.e., a serious incident should not occur if these conditions are present).
Remains effective even in the case of unintentional human error unrelated to the installation of the control.
Examples of direct controls include:

Lockout/Tagout (LOTO)
Machine guarding
Hard physical barriers
Fall protection systems
Covers or shields
Examples that are NOT direct controls include:

Training
Warning signs
Rules and procedures
Experience
Standard personal protective equipment (e.g., hard hats, gloves, boots)
Does the example provided contain a direct control as defined above? Direct Control Answer: [Answer]

Provide your answers in this format:  yes or  no and add a justification
1. [Your answer]
2. [Your answer]
3. [Your answer]
4. [Your answer]
"""

USER_PROMPT_FORMAT = """
Field notes:
{field_notes}

Respond strictly with  yes or 'no' for each question and a justification, one answer per line, in this format:
1 will be high energy, 2 will be high energy incident, 3 will be serious injury, 4 will be direct control
1. [your answer]
2. [your answer]
3. [your answer]
4. [your answer]

Here are some examples...
"""



def classify_risk(notes):
    prompt_field_notes = notes

    output = ollama.chat(
        model='llama3:latest',
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': USER_PROMPT_FORMAT.format(field_notes=prompt_field_notes)}
        ],
        options={'temperature': 0}
    )

    response = output['message']['content']
    print(f"Raw Response: {response}")

    if not response.strip():
        print("Model returned an empty response.")
        return 'Unable to classify'

    ans = [line.split('. ')[1].lower().strip() for line in response.split('\n') if line.strip() and '. ' in line]

    print(f"Parsed Answers: {ans}")

    if len(ans) >= 4:
        if ans[0] == 'yes' and ans[1] == 'yes' and ans[2] == 'yes':
            return 'HSIF'
        elif ans[0] == 'yes' and ans[1] == 'yes' and ans[2] == 'no' and ans[3] == 'no':
            return 'PSIF'
        elif ans[0] == 'no' and ans[2] == 'yes':
            return 'LSIF'
        elif ans[0] == 'yes' and ans[1] == 'no' and ans[3] == 'yes':
            return 'success'
        elif ans[0] == 'yes' and ans[1] == 'no' and ans[3] == 'no':
            return 'exposure'
        elif ans[0] == 'yes' and ans[1] == 'yes' and ans[2] == 'no' and ans[3] == 'yes':
            return 'capacity'
        else:
            return 'low severity'
    else:
        print(f"Expected 4 answers but got {len(ans)}: {ans}")
        return 'Unable to classify'


# Apply the classification to each row and print results
for index, row in df.iterrows():
    notes = row[['PNT_ATRISKNOTES_TX', 'PNT_ATRISKFOLWUPNTS_TX']].to_frame().to_string(index=True, header=False, max_colwidth=None)        
    classification = classify_risk(notes)
    print(f"Index: {index}")
    print(f"Notes: {notes}")  # Print first 100 characters of notes
    print(f"Classification: {classification}")
    print("---")

print("Classification complete. Results printed above.")