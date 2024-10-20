import ollama
import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('data.csv')
df = df.sample(n=10, random_state=42)  # Set random_state for reproducibility
df = pd.DataFrame({
    'PNT_ATRISKNOTES_TX': [
        "An employee was on the top of a de-energized transformer at 25 feet of height with a proper fall arrest system. While working, she tripped on a lifting lug, falling within 2 feet from an unguarded edge. When the employee landed, she sprained her wrist.",
        "An employee contracted West Nile Virus after being bitten by a mosquito while at work in a boggy area. Because of the exposure, the employee was unconscious and paralyzed for a two-week period.",
        "An employee was working alone and placed an extension ladder against the wall. When he reached 10 feet of height, the ladder feet slid out and he fell with the ladder to the floor. The employee was taken to the hospital for a bruise to his right leg and remained off duty for three days.",
        "A crew was closing a 7-ton door on a coal crusher. As the door was lowered, an observer noticed that the jack was not positioned correctly and could tip. The observer also noted that workers were nearby, within 4 feet of the jack.",
        "Workers were hoisting beams and steel onto a scaffold. A certified mechanic operated an air hoist to lift the beam. As the lift was performed, the rigging was caught under an adjacent beam. Under the increasing tension, the cable snapped and struck a second employee in the leg, fully fracturing his femur. An investigation indicated that the rigging was not properly inspected before the lift.",
        "A dozer was operating on a pet coke pile and slid down an embankment onto the cab after encountering a void in the pile. The operator was wearing his seat belt, and the roll cage kept the cab from crushing. No workers or machinery were nearby, and no injuries were sustained.",
        "A master electrician was called to work on a new 480-volt service line in a commercial building. When working on the meter cabinet, the master electrician had to position himself awkwardly between the cabinet and a standpipe. He was not wearing an arc-rated face shield, balaclava, or proper gloves. During the work, an arc flash occurred, causing third-degree burns to his face.",
        "An employee was descending a staircase and when stepping down from the last step she rolled her ankle on an extension cord on the floor. She suffered a torn ligament and a broken ankle that resulted in persistent pain for more than a year.",
        "A crew was working near a sedimentation pond on a rainy day. The boom of the trac-hoe was within 3 feet of a live 12kV line running across the road. No contact was made because a worker intervened and communicated with the operator.",
        "A crew was working in a busy street to repair a cable fault. During the work, the journeyman took a step back from the truck outside of the protected work zone into oncoming traffic. A driver slammed on his brakes and stopped within one foot of the journeyman. No injuries were sustained."
    ],
    'PNT_ATRISKFOLWUPNTS_TX': ["NA"] * 10  # No follow-up notes for all test cases
}, index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# System prompts for classification
SYSTEM_PROMPT_PART1 = """
Based on the following patient notes, please answer the questions with 'yes' or 'no' only.

High Energy Criteria: Suspended Loads: Loads greater than 500 lbs lifted more than 1 foot off the ground. Elevation: Any height exceeding 4 feet, considering the average weight of a human (over 150 lbs). Mobile Equipment: Most mobile equipment, including motor vehicles, exceeds the high-energy threshold when in motion from the perspective of a worker on foot. Work Zone Traffic: An incident occurs if a vehicle departs its intended path within 6 feet of an exposed employee or if an employee enters the traffic pattern. Motor Vehicle Speed: An estimate of 30 miles per hour is considered the high-energy threshold for serious or fatal crashes. Mechanical Energy: Heavy rotating equipment beyond powered hand tools typically exceeds the high-energy threshold. Temperature: Exposure to substances at or above 150 degrees Fahrenheit can cause third-degree burns if contacted for 2 seconds or more. Any release of steam or combustion materials (e.g., paper burning at approximately 700 degrees Fahrenheit) exceeds the high-energy threshold. Explosions: Any incident described as an explosion exceeds the high-energy threshold. Unsupported Soil: Soil in a trench or excavation exceeding 5 feet of height, with pressure increasing approximately 40 pounds per square foot for each foot of depth. Electrical Energy: Voltage equal to or exceeding 50 volts can result in serious injury or death. Any arc flash also exceeds the high-energy threshold. Toxic Chemicals or Radiation: Use IDLH (Immediately Dangerous to Life or Health) values from the CDC and consider: Oxygen (O2) levels below 16% Corrosive chemical exposures (pH <2 or >12.5) Additional Considerations: If a situation does not fit any of the above criteria but encompasses potential energy: Estimate the weight in pounds (lbs) and height in feet (ft). If height is greater than 503.1 × weight − 0.99 503.1×weight −0.99 , it can be classified as high energy. If it encompasses kinetic energy: Estimate the weight in pounds (lbs) and speed in miles per hour (mph). If speed is greater than 𝑦 = 182.71 × weight − 0.68 , it can be classified as high energy.


Criteria Answer: [Answer]
"""

SYSTEM_PROMPT_PART2 = """
High-Energy Incident:
A high-energy incident is defined as an occurrence where:

A high-energy source is released, changing state and becoming hazardous in the work environment.
The energy source must be no longer contained or under the control of the worker.
The worker must have either:
Contact: direct transmission of high energy to the human body.
Proximity: within 6 feet of the energy source with unrestricted egress, or any distance in confined spaces or situations where escape is restricted.
Does the event described meet the criteria for a high-energy incident? Incident Answer: [Answer]
"""

SYSTEM_PROMPT2 = """
Severe Injury:
Is there a severe injury in the example? Severe Injury Answer: [Answer]
"""

SYSTEM_PROMPT3 = """
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
"""

def classify_risk(notes):
    prompt_field_notes = notes

    output = ollama.chat(
        model='llama3:latest',
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT_PART1},
            {'role': 'system', 'content': SYSTEM_PROMPT_PART2},
            {'role': 'system', 'content': SYSTEM_PROMPT2},
            {'role': 'system', 'content': SYSTEM_PROMPT3},
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