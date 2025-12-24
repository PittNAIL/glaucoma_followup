#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import pandas as pd
import glob
import argparse
from collections import Counter
import logging

# --- Configuration Constants ---

# Time conversion factors
DAYS_PER_WEEK = 7
DAYS_PER_MONTH = 30.4375  # Average days in a month (365.25 / 12)
DAYS_PER_YEAR = 365.25

# Standardized unit names
UNIT_DAY = 'day'
UNIT_WEEK = 'week'
UNIT_MONTH = 'month'
UNIT_YEAR = 'year'
UNIT_MISC = 'misc'

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Regex Definitions for Follow-up Time ---

# Preposition pattern for flexibility (in, within, about, around)
PREP_PATTERN = r'(?:(?:in|within|about|around)\s+)?'

# Priority 1: High-confidence follow-up keywords (more tolerant)
regRTC = rf'\b(?:rtc|return to clinic)\b(?:\s+\w+){{0,4}}?\s*{PREP_PATTERN}(\d+(?:\.\d{{1,2}})?)\s*([a-zA-Z]+)'
regReturn = rf'\b(?:return|rv)\b(?:\s+\w+){{0,4}}?\s*{PREP_PATTERN}(\d+(?:\.\d{{1,2}})?)\s*([a-zA-Z]+)'

# Priority 2: Standard follow-up keywords
regFollowup = rf'(?:follow[\s-]?up|followup|f/?u\.?|F/?U\.?)[\s:]*{PREP_PATTERN}(\d+(?:\.\d{{1,2}})?)\s*([a-zA-Z]+)'

# Priority 3: Review/recheck/repeat/revisit (any suffix form)
regReview = rf'(?:review\w*|recheck\w*|repeat\w*|revisit\w*)[\s:]*{PREP_PATTERN}(\d+(?:\.\d{{1,2}})?)\s*([a-zA-Z]+)'

# Priority 4: See back
regSeeBack = rf'(?:see\s+(?:me|you)|come\s+back|back\s+in)[\s:]*(\d+(?:\.\d{{1,2}})?)\s*([a-zA-Z]+)'

# Priority 5: Next visit / retina follow-up
regNextVisit = rf'\b(?:NV|Next Visit|f/u with retina|retina in)\s+{PREP_PATTERN}(\d+(?:\.\d{{1,2}})?)\s*([a-zA-Z]+)'

# Priority 6: Refer to subspecialty in X time
regRefer = rf'\brefer(?:red)?\s+to\s+\w+(?:\s+\w+)*\s+{PREP_PATTERN}(\d+(?:\.\d{{1,2}})?)\s*([a-zA-Z]+)'

# Priority 7: Time ranges (extracts larger value)
regTimeRange = r'(\d+(?:\.\d{1,2})?)[\s-]+(?:to|or)?\s*(\d+(?:\.\d{1,2})?)\s*([a-zA-Z]+)'

# Priority 10: Special keywords
regAnnual = r'\b(?:annual(?:ly)?|yearly)\b'
regBiannual = r'\b(?:biannual(?:ly)?|twice\s+yearly)\b'
regMonthly = r'\b(?:monthly)\b'

# Priority 200: PRN
regPRN = r'\b(?:prn|as\s+needed|next\s+available)\b'

# --- Regex Definitions for Exclusions ---

regAge = r'\b(\d{1,2})\s*year[s]?\s*old\b'
regPastVisitContext = r'(?:returns?|returned|here|presents?|presented|comes?|came|seen|visit|evaluation|f/?u)\s+(?:for|s/p|at|was|last)\s'
regPastVisit = rf'(?:{regPastVisitContext}|s/p|post-op|pow\s+\d+)\s*.*?(\d+)\s*([a-zA-Z]+)\s*(?:follow[\s-]?up|visit|f/?u|testing|exam|check|recheck)'
regPastRecheck = r'(\d+)\s*([a-zA-Z]+)\s+(?:recheck|check|exam|visit|f/?u|follow[\s-]?up)\s+[A-Z\[]'
regCurrentVisitStart = r'^(?:[\s\d/-]+)?(\d+)\s*([a-zA-Z]+)\s+f/?u\b'
regPastAnnual = r'(?:presents? for|here for|s/p)\s+(?:an\s+)?(?:annual|yearly)\s+(?:eval|exam|visit)'
regMedFrequency = r'(?:q|Q|every|for|x)\s*(\d+)\s*(days?|weeks?|months?)'
# New: exclude past tense forms unless followed by "in X"
regPastReviewed = r'\b(?:reviewed|rechecked|repeated|revisited)\b(?!\s+in\s+\d)'

# --- Glaucoma detection regex (unchanged, except more tolerant with glaucomatous)
regGlaucomaPrimary = r'\b(?:glaucoma|glaucomatous|JOAG|POAG|OAG|NTG|CACG|NVG|pigmentary glaucoma|pseudoexfoliation glaucoma|steroid.{0,10}induced glaucoma|trabeculectomy|tube shunt|ahmed valve|baerveldt|express shunt|SLT|selective laser trabeculoplasty|argon laser trabeculoplasty)\b'
regPressure = r'\b(?:iop\b|intraocular pressure|eye pressure|ocular hypertension|OHTN)'
regOpticNerve = r'\b(?:optic nerve|optic disc|optic cup|CDR|cup.{0,10}disc|RNFL|ganglion cell|nerve fiber layer|cupping|excavation)\b'
regVisualField = r'\b(?:visual field|perimetry|HVF|field defect|scotoma)\b'

# --- Helper Functions ---

def normalize_unit(unit_str):
    """Normalize time unit variants into standard form."""
    unit_str = str(unit_str).lower().strip().rstrip('.')
    if unit_str in ['d', 'day', 'days']:
        return UNIT_DAY
    elif unit_str in ['wk', 'wks', 'week', 'weeks']:
        return UNIT_WEEK
    elif unit_str in ['mo', 'mos', 'month', 'months']:
        return UNIT_MONTH
    elif unit_str in ['yr', 'yrs', 'year', 'years']:
        return UNIT_YEAR
    return None

def convert_to_days(num, unit):
    """Convert intervals to days."""
    try:
        num = float(num)
    except (ValueError, TypeError):
        return None
    if unit == UNIT_DAY:
        return num
    elif unit == UNIT_WEEK:
        return num * DAYS_PER_WEEK
    elif unit == UNIT_MONTH:
        return num * DAYS_PER_MONTH
    elif unit == UNIT_YEAR:
        return num * DAYS_PER_YEAR
    return None

def get_exclusion_spans(text):
    exclusion_spans = []
    exclusion_patterns = [regAge, regPastVisit, regPastRecheck, regCurrentVisitStart,
                          regMedFrequency, regPastAnnual, regPastReviewed]
    for pattern in exclusion_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            exclusion_spans.append((match.start(), match.end()))
    return exclusion_spans

# --- Extraction Logic ---

def extract_followup(note_text):
    all_times = []
    exclusion_spans = get_exclusion_spans(note_text)

    def is_excluded(match_obj):
        for start, end in exclusion_spans:
            if max(match_obj.start(), start) < min(match_obj.end(), end):
                return True
        return False

    inclusion_patterns = [
        (regRTC, 1, 'RTC'),
        (regReturn, 2, 'Return'),
        (regFollowup, 3, 'Followup'),
        (regNextVisit, 4, 'NextVisit'),
        (regReview, 5, 'Review'),
        (regSeeBack, 6, 'SeeBack'),
        (regRefer, 7, 'Refer')
    ]

    for pattern, priority, source in inclusion_patterns:
        for match in re.finditer(pattern, note_text, re.IGNORECASE):
            if is_excluded(match): continue
            time_val, unit = match.group(1), normalize_unit(match.group(2))
            if unit:
                days = convert_to_days(time_val, unit)
                if days:
                    all_times.append({'time': time_val, 'unit': unit, 'days': days,
                                      'priority': priority, 'source': source, 'match_start': match.start()})

    for match in re.finditer(regTimeRange, note_text, re.IGNORECASE):
        if is_excluded(match): continue
        context_window = note_text[max(0, match.start()-80):match.start()].lower()
        context_keywords = ['rtc','return','return to','f/u','follow-up','followup','see back','schedule','refer to']
        if any(k in context_window for k in context_keywords):
            try:
                val1, val2 = float(match.group(1)), float(match.group(2))
                unit = normalize_unit(match.group(3))
                if unit:
                    time_val = str(max(val1, val2))
                    days = convert_to_days(time_val, unit)
                    all_times.append({'time': time_val, 'unit': unit, 'days': days,
                                      'priority': 8, 'source': 'TimeRange', 'match_start': match.start()})
            except ValueError:
                continue

    # Specials
    if re.search(regAnnual, note_text, re.IGNORECASE):
        all_times.append({'time':'1','unit':UNIT_YEAR,'days':DAYS_PER_YEAR,'priority':10,'source':'Annual','match_start':0})
    if re.search(regBiannual, note_text, re.IGNORECASE):
        all_times.append({'time':'6','unit':UNIT_MONTH,'days':DAYS_PER_MONTH*6,'priority':10,'source':'Biannual','match_start':0})
    if re.search(regMonthly, note_text, re.IGNORECASE):
        all_times.append({'time':'1','unit':UNIT_MONTH,'days':DAYS_PER_MONTH,'priority':10,'source':'Monthly','match_start':0})
    if re.search(regPRN, note_text, re.IGNORECASE):
        all_times.append({'time':'PRN','unit':UNIT_MISC,'days':float('inf'),'priority':200,'source':'PRN','match_start':0})

    if not all_times:
        return None, None

    # Deduplicate + prioritize
    unique = {}
    for t in sorted(all_times, key=lambda x: (x['priority'], x['match_start'])):
        key = (t['time'], t['unit'])
        if key not in unique:
            unique[key] = t
    filtered = list(unique.values())
    non_prn = [t for t in filtered if t['unit'] != UNIT_MISC]

    if non_prn:
        best = sorted(non_prn, key=lambda x: (x['priority'], -x['days']))[0]
    else:
        best = filtered[0]
    return best['time'], best['unit']

# --- Glaucoma detection ---

def extract_glaucoma_hit(note_text):
    negs = ['denies','no history of','negative for','no evidence of','rule out','r/o','suspicion for']
    for match in re.finditer(regGlaucomaPrimary, note_text, re.IGNORECASE):
        window = note_text[max(0, match.start()-40):match.start()].lower()
        if not any(n in window for n in negs):
            return match.group(0)
    pt1, pt2, pt3 = re.search(regPressure, note_text, re.IGNORECASE), \
                    re.search(regOpticNerve, note_text, re.IGNORECASE), \
                    re.search(regVisualField, note_text, re.IGNORECASE)
    if (pt1 and pt2) or (pt1 and pt3) or (pt2 and pt3):
        return "+".join([x.group(0) for x in [pt1,pt2,pt3] if x])
    return None

def annotate_note(note_text):
    fu_time, fu_unit = extract_followup(note_text)
    hit = extract_glaucoma_hit(note_text)
    return {
        'GLAUCOMA_HIT': 'Y' if hit else 'N',
        'HIT_WORD': hit if hit else 'N/A',
        'FOLLOWUP_TIME': fu_time if fu_time else '',
        'FOLLOWUP_UNIT': fu_unit if fu_unit else ''
    }

def process_dataframe(df):
    results = [annotate_note(r['NOTE_TEXT']) for _,r in df.iterrows()]
    ann = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), ann.reset_index(drop=True)], axis=1)

def load_notes_to_dataframe(notes_directory):
    txt_files = glob.glob(os.path.join(notes_directory, "*.txt"))
    data = []
    for filepath in txt_files:
        base = os.path.basename(filepath)
        name = os.path.splitext(base)[0]
        parts = name.split('_')
        if len(parts) >= 3:
            study_id, contact_date, note_id = parts[0], parts[1], '_'.join(parts[2:])
        else:
            continue
        try:
            with open(filepath,'r',encoding='utf-8') as f:
                text = f.read().strip()
        except: continue
        if len(text) < 20: continue
        data.append({'STUDY_ID':study_id,'CONTACT_DATE':contact_date,'NOTE_ID':note_id,
                     'NOTE_TEXT':text,'FILENAME':base,'NOTE_LENGTH':len(text)})
    return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser(description="Clinical Note Annotation Script for Glaucoma and Follow-up Time")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        logging.error("Input directory not found.")
        return
    df = load_notes_to_dataframe(args.input_dir)
    if df.empty:
        logging.error("No notes loaded.")
        return

    df_ann = process_dataframe(df)
    df_ann.rename(columns={'FOLLOWUP_TIME':'time1','FOLLOWUP_UNIT':'date',
                           'GLAUCOMA_HIT':'hit','HIT_WORD':'hit word'}, inplace=True)
    df_ann['time2'] = ''
    df_ann['agree FU'] = 'N/A'
    df_ann['agree Hit'] = 'N/A'
    final_cols = ['STUDY_ID','CONTACT_DATE','NOTE_ID','NOTE_TEXT','FILENAME','NOTE_LENGTH',
                  'time1','date','time2','hit','hit word','agree FU','agree Hit']
    df_final = df_ann[[c for c in final_cols if c in df_ann.columns]]
    df_final.to_csv(args.output_csv,index=False)
    logging.info(f"Saved annotated data to {args.output_csv}")
    logging.info(f"Glaucoma hits: {df_final['hit'].value_counts().get('Y',0)}")
    logging.info(f"Follow-up times detected: {sum(df_final['time1']!='')}")

if __name__ == "__main__":
    main()
