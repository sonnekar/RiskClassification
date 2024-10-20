from flask import Flask, render_template, request
import pandas as pd
import ollama

app = Flask(__name__)

data_columns = ['OBSRVTN_NB', 'DATETIME_DTM', 'PNT_NM', 'QUALIFIER_TXT', 'PNT_ATRISKNOTES_TX', 'PNT_ATRISKFOLWUPNTS_TX']
data_frame = pd.DataFrame(columns=data_columns)

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

Respond strictly with yes or 'no' for each question and a justification, one answer per line, in this format:
1 will be high energy, 2 will be high energy incident, 3 will be serious injury, 4 will be direct control
1. [your answer]
2. [your answer]
3. [your answer]
4. [your answer]
"""

def classify_risk(notes):
    output = ollama.chat(
        model='llama3:latest',
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT_PART1},
            {'role': 'system', 'content': SYSTEM_PROMPT_PART2},
            {'role': 'system', 'content': SYSTEM_PROMPT2},
            {'role': 'system', 'content': SYSTEM_PROMPT3},
            {'role': 'user', 'content': USER_PROMPT_FORMAT.format(field_notes=notes)}
        ],
        options={'temperature': 0}
    )

    response = output['message']['content']
    if not response.strip():
        return 'Unable to classify'

    ans = [line.split('. ')[1].lower().strip() for line in response.split('\n') if line.strip() and '. ' in line]

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
        return 'Unable to classify'

@app.route('/', methods=['GET', 'POST'])
def index():
    global data_frame  

    if request.method == 'POST':
        obsrvtn_nb = request.form['OBSRVTN_NB']
        datetime_dtm = request.form['DATETIME_DTM']
        pnt_nm = request.form['PNT_NM']
        qualifier_txt = request.form['QUALIFIER_TXT']
        pnt_atrisknotes_txt = request.form['PNT_ATRISKNOTES_TX']
        pnt_atriskfolwupnts_txt = request.form['PNT_ATRISKFOLWUPNTS_TX']

        new_data = pd.DataFrame([{
            'OBSRVTN_NB': obsrvtn_nb,
            'DATETIME_DTM': datetime_dtm,
            'PNT_NM': pnt_nm,
            'QUALIFIER_TXT': qualifier_txt,
            'PNT_ATRISKNOTES_TX': pnt_atrisknotes_txt,
            'PNT_ATRISKFOLWUPNTS_TX': pnt_atriskfolwupnts_txt
        }])

        data_frame = pd.concat([data_frame, new_data], ignore_index=True)

        expected_columns = ['PNT_ATRISKNOTES_TX', 'PNT_ATRISKFOLWUPNTS_TX']
        if set(expected_columns).issubset(data_frame.columns):
            notes = "\n".join(data_frame[expected_columns].astype(str).values.flatten())
            classification = classify_risk(notes)
        else:
            classification = "Columns not found in the DataFrame."

        return f"""
            <h2>Submitted Data</h2>
            <ul>
                <li>OBSRVTN_NB: {obsrvtn_nb}</li>
                <li>DATETIME_DTM: {datetime_dtm}</li>
                <li>PNT_NM: {pnt_nm}</li>
                <li>QUALIFIER_TXT: {qualifier_txt}</li>
                <li>PNT_ATRISKNOTES_TX: {pnt_atrisknotes_txt}</li>
                <li>PNT_ATRISKFOLWUPNTS_TX: {pnt_atriskfolwupnts_txt}</li>
                <li>Classification: {classification}</li>
            </ul>
        """
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5001)
