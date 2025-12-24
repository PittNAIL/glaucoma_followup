#!/usr/bin/env python3
"""
Glaucoma Follow-up Annotation Script
Processes text files and annotates them for glaucoma detection and follow-up times
Compatible with the evaluation script format
"""

import os
import re
import pandas as pd
import glob
import argparse
from collections import Counter

# ============================================================================
# REGEX PATTERNS
# ============================================================================

# Follow-up keywords strings
regAfter = r'([^.;|]{,1})((monitor\b)|(monitored.{,10}every))(?=([^;|]{,25}))'
regBefore = r'([^.;|]{,10})((with testing)|(testing.{,10}hvf)|(sooner.{,5}if worse))(?=([^;|]{,1}))'
regMed = r'([^.;|]{,14})((timolol|latanoprost|brimonidine|dorzolamide|travoprost|bimatoprost|combigan|cosopt|xalatan|lumigan|alphagan|trusopt|azopt|rhopressa|vyzulta|netarsudil|latanoprostene|SLT|ALT|trabeculectomy|tube shunt|ahmed valve|baerveldt|express shunt|(\boct\b)|HVF|\bprp\b|iop check|\biop\b|\bvf\b|\bhrt\b|7 fields color|Spectralis|tonometry|visual field|perimetry|gonioscopy))(?=([^;|(]{,40}))'
regDil = r'([^.;|]{,12})((\bdil\b)|(dilation)|dilated?|\bdfe\b|dilated fundus exam)(?=([^;|]{,25}))'
regAppt = r'([^.;|]{,14})(appointment|(re)?evaluation|examination|(exam\b)|examine|repeat.{,20}exam|repeat testing|next visit|will repeat|glaucoma exam)(?=([^;|]{,35}))'
regAmbig = r'([^.;,|]{,8})((back in)|repeat in|refer to.{,15}within|recheck.{,10}in|re-?evaluate|will.{,10}see|see me|will see|come back|recommend(ed)?|\bdue\b.{,10}in|check|\brx\b|extend to|review.{,15}in|see\b.{,15}\bin|surveillance|reassess)(?=([^;|]{,25}))'
regKey2 = r'([^.;)|]{,15})(follow(ed)?[ -]?(up)?\b|follow-up|followup|\bf/?u\b|follow up with|Follow-  up)(?=([^;|]{,50}))'
regKey = r'([^.;)|]{,8})((\brto\b)|(\brtn\b)|\bret\b|return to clinic|return to Dr|return|\brtc\b)(?=([^;|]{,35}))'

# Glaucoma follow-up terms
regGlaucoma_FU = r'([^.;)]{,10})(glaucoma|(\biop\b)|intraocular pressure|visual field|perimetry|HVF|optic nerve|optic disc|cup.{,10}disc ratio|CDR|RNFL|ganglion cell|OCT.{,10}RNFL|pressure.{,10}control|target.{,10}pressure|\bIOP\b.{,10}control|trabeculectomy|tube shunt|SLT|ALT|selective laser|argon laser)(?=([^;|]{,50}))'

# Priorities
regSearches = [regKey, regKey2, regDil, regAfter, regAmbig, regMed, regBefore, regAppt]

# Glaucoma Detection Patterns
regGlaucoma = r'(?P<pre>[^.;]{,4})(?P<glaucoma>glaucoma|JOAG|(\biop\b).{,15}(elevated|high|increased)|intraocular pressure.{,15}(elevated|high|mmhg)|pressure.{,10}(control|elevated|high)|visual field.{,10}(defect|loss|damage)|optic nerve.{,10}(damage|cupping|pallor)|optic disc.{,10}(cupping|excavation|pallor)|cup.{,10}disc.{,10}ratio|\bCDR\b.{,10}(enlarged|increased)|RNFL.{,10}(thinning|loss|damage)|ganglion cell.{,10}(loss|damage)|target.{,10}pressure|\bIOP\b|trabeculectomy|tube shunt|ahmed valve|baerveldt|express shunt|SLT|ALT|selective laser trabeculoplasty|argon laser trabeculoplasty|narrow.{,10}angle|angle.{,10}closure|pigment.{,10}dispersion|pseudoexfoliation|steroid.{,10}induced|secondary glaucoma|open.{,10}angle|closed.{,10}angle)(?P<post>[^.;]{,4})'

# Separate patterns for modular search
regPressure = r'((\biop\b)|intraocular pressure|eye pressure|ocular pressure|pressure.{,10}control|target.{,10}pressure)'
regOpticNerve = r'(optic nerve|optic disc|optic cup|CDR|cup.{,10}disc|RNFL|ganglion cell|nerve fiber layer)'
regVisualField = r'(visual field|perimetry|HVF|field defect|scotoma|visual loss)'
regTreatment = r'(trabeculectomy|tube shunt|ahmed|baerveldt|SLT|ALT|laser trabeculoplasty|glaucoma drops|timolol|latanoprost|brimonidine|dorzolamide|cosopt|alphagan|travatan)'

# Stable glaucoma terms
regStab = r'(stable|pressure.{,10}controlled|IOP.{,10}controlled|glaucoma.{,10}stable|visual field.{,10}stable|optic nerve.{,10}stable|target.{,10}pressure|well.{,10}controlled|monitor|surveillance|routine|follow.{,10}up)'

# Date Keywords
regYear = r'((?P<t>\d+(\.\d)?|next|this|every|each|one|two|three|four|five|six|seven|eight|nine|ten|\ba\b)(\W|to){,3}(?P<t2>(\b\d+)\D{,2})?(years?|yrs?\b))([^.;]{,8})'
regMonth = r'((?P<t>\d+(\.\d)?|this|several|next|each|one|few|two|three|four|five|six|seven|eight|nine|ten|in the|within the|in a|after a|next|\ba\b)([^a-z0123456789:]|to){,3}(?P<t2>(\b\d+)\D{,2})?(mm?o?o?n?ths?|mos?\b|m\b))([^.;]{,8})'
regWeek = r'((?P<t>\d+(\.\d)?|this|couple|next|few|one|two|three|four|five|six|seven|eight|nine|ten|\ba\b)(\W|to){,3}(?P<t2>(\b\d+|\btwo\b|\bthree\b|\bfour\b|\bfive\b|\bsix\b|\bseven\b|\beight\b|\bnine\b)\D{,2})?(weeks?|ws?\b|wks?\b))([^.;]{,8})'
regDay = r'((?P<t>\d+(\.\d)?|next|few|one|two|three|four|five|six|seven|eight|nine|ten)(\W|to){,3}(?P<t2>(\b\d+)[^/\d]{,2})?(\bdays?))([^.;]{,8})'
regExtra = r'annual(ly)?|yearly'
regExtraBi = r'biannual(ly)?|biyearly|twice yearly'
regExtraWeek = r'(this|next).{,5}(monday|tuesday|wednesday|thursday|friday|\bsat(urday)?\b)'
regExtraDay = r'tomorrow'
regExtraMonth = r'monthly'
regDate = [regDay, regWeek, regMonth, regYear]

# Misc
regMisc = r'(?P<note>(prn|as needed|next available))'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def word_to_number(word):
    """Convert word numbers to integers"""
    word_to_num = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'a': 1, 'an': 1, 'next': 1, 'this': 1, 'few': 2
    }
    return word_to_num.get(word.lower(), word)

def findDate(search_text):
    """Extract date information from search text"""
    numTag = numTag2 = dateTag = None
    
    if not search_text:
        return numTag, numTag2, dateTag
    
    # Check for special cases first
    misc_match = re.search(regMisc, search_text, re.IGNORECASE)
    if misc_match:
        return "PRN", None, "misc"
    
    extra_match = re.search(regExtra, search_text, re.IGNORECASE)
    if extra_match:
        return "1", None, "year"
    
    extra_bi_match = re.search(regExtraBi, search_text, re.IGNORECASE)
    if extra_bi_match:
        return "6", None, "month"
    
    extra_month_match = re.search(regExtraMonth, search_text, re.IGNORECASE)
    if extra_month_match:
        return "1", None, "month"
    
    # Check date patterns
    for i, pattern in enumerate(regDate):
        match = re.search(pattern, search_text, re.IGNORECASE)
        if match:
            t = match.group('t') if 't' in match.groupdict() else None
            t2 = match.group('t2') if 't2' in match.groupdict() else None
            
            if t:
                try:
                    numTag = str(int(t))
                except ValueError:
                    converted = word_to_number(t)
                    if isinstance(converted, int):
                        numTag = str(converted)
                    else:
                        numTag = t
            
            if t2:
                try:
                    numTag2 = str(int(t2))
                except ValueError:
                    numTag2 = t2
            
            dateTag = ["day", "week", "month", "year"][i]
            break
    
    return numTag, numTag2, dateTag

def load_notes_to_dataframe(notes_directory):
    """Load all text files into a pandas DataFrame"""
    txt_files = glob.glob(os.path.join(notes_directory, "*.txt"))
    print(f"Found {len(txt_files)} text files to load")
    
    data = []
    for filepath in txt_files:
        # Parse filename for metadata
        basename = os.path.basename(filepath)
        name_without_ext = os.path.splitext(basename)[0]
        parts = name_without_ext.split('_')
        
        if len(parts) >= 3:
            study_id = parts[0]
            contact_date = parts[1]
            note_id = '_'.join(parts[2:])  # In case note_id has underscores
        else:
            print(f"Warning: Could not parse filename {basename}")
            continue
        
        # Read note text
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                note_text = f.read().strip()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue
        
        if len(note_text) < 10:  # Skip very short notes
            continue
            
        data.append({
            'STUDY_ID': study_id,
            'CONTACT_DATE': contact_date,
            'NOTE_ID': note_id,
            'NOTE_TEXT': note_text,
            'FILENAME': basename,
            'NOTE_LENGTH': len(note_text)
        })
    
    df = pd.DataFrame(data)
    print(f"Successfully loaded {len(df)} notes into DataFrame")
    return df

def annotate_glaucoma_note(note_text):
    """Annotate a single note for glaucoma and follow-up"""
    
    # Initialize return values
    glaucoma_hit = False
    hit_word = ''
    followup_time = followup_time2 = followup_unit = None
    
    # Get end of note for initial search
    if len(note_text) > 85:
        end = note_text.strip()[-85:]
    else:
        end = note_text
    
    # Follow-up detection
    search = ""
    
    # First search of the end
    for x in range(len(regSearches)):        
        if followup_unit:            
            break
        raw = re.findall(regSearches[x], end, re.IGNORECASE)
        for hit in raw:
            search += hit[0] + "| " + hit[4] + "| "                        
        if raw:
            time1, time2, unit = findDate(search)
            if unit:
                followup_time, followup_time2, followup_unit = time1, time2, unit
                if followup_time:
                    try:
                        if int(followup_time) > 12 or (int(followup_time) > 1 and "year" in str(followup_unit)):            
                            followup_time = followup_time2 = followup_unit = None
                    except:
                        pass
    
    # Second search of full text if no date found
    if not followup_unit:            
        for x in range(len(regSearches)):
            search = ""
            if followup_unit:
                break
            raw = re.findall(regSearches[x], note_text, re.IGNORECASE)
            for hit in raw:
                search += hit[0] + "| " + hit[4] + "| "                                    
            if raw:
                time1, time2, unit = findDate(search)
                if unit:
                    followup_time, followup_time2, followup_unit = time1, time2, unit
                    if followup_time:
                        try:
                            if int(followup_time) > 12 or (int(followup_time) > 1 and "year" in str(followup_unit)):            
                                followup_time = followup_time2 = followup_unit = None
                        except:
                            pass

    # Third search of just the end regardless of keywords
    if not followup_unit:
        regEnd = r'(?P<e>[^)]{,15}$)'
        end_match = re.search(regEnd, note_text.strip(), re.IGNORECASE)
        if end_match:
            search = end_match.group('e')
            time1, time2, unit = findDate(search)
            if unit:
                followup_time, followup_time2, followup_unit = time1, time2, unit
    
    # Glaucoma detection
    # Main glaucoma search
    GlaucomaSearch = re.findall(regGlaucoma, note_text, re.IGNORECASE)
    for hit in GlaucomaSearch:        
        if len(hit) > 2 and hit[2]:            
            test = hit[2]
            if test[0].isupper() and (len(test) < 2 or test[1].islower()):
                continue
            else:
                glaucoma_hit = True
                hit_word = hit[1]
                break
        elif len(hit) > 1:
            glaucoma_hit = True
            hit_word = hit[1]            
            break
    
    # Secondary search: if no hits, separate searches for pressure, optic nerve, and visual field
    if not glaucoma_hit: 
        pt1 = re.search(regPressure, note_text, re.IGNORECASE)
        pt2 = re.search(regOpticNerve, note_text, re.IGNORECASE)
        pt3 = re.search(regVisualField, note_text, re.IGNORECASE)
        if (pt1 and pt2) or (pt1 and pt3) or (pt2 and pt3):
            hit_word_parts = []
            if pt1: hit_word_parts.append(pt1.group(0))
            if pt2: hit_word_parts.append(pt2.group(0))
            if pt3: hit_word_parts.append(pt3.group(0))
            hit_word = "+".join(hit_word_parts)
            glaucoma_hit = True
    
    # Third search: stable terms (if still no hits)
    if not glaucoma_hit:
        stab = re.search(regStab, note_text, re.IGNORECASE)
        if stab:
            glaucoma_hit = True
            hit_word = stab.group(0)
    
    return {
        'GLAUCOMA_HIT': 'Y' if glaucoma_hit else 'N',
        'HIT_WORD': hit_word if hit_word else '',
        'FOLLOWUP_TIME': followup_time if followup_time else '',
        'FOLLOWUP_TIME2': followup_time2 if followup_time2 else '',
        'FOLLOWUP_UNIT': followup_unit if followup_unit else ''
    }

def process_notes_dataframe(df):
    """Process the entire DataFrame and add annotation columns"""
    print("Processing notes for glaucoma annotation...")
    
    # Apply annotation to each note
    annotations = df['NOTE_TEXT'].apply(annotate_glaucoma_note)
    
    # Convert to DataFrame and join with original
    annotation_df = pd.DataFrame(annotations.tolist())
    result_df = pd.concat([df, annotation_df], axis=1)
    
    return result_df

def main():
    """Main processing function"""
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description='Annotate clinical notes for glaucoma and follow-up times',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-i', '--input', 
                        type=str,
                        required=True,
                        help='Directory containing text files to process')
    
    parser.add_argument('-o', '--output', 
                        type=str,
                        required=True,
                        help='Output CSV file path')
    
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    notes_directory = os.path.expanduser(args.input)
    output_file_path = os.path.expanduser(args.output)
    
    # Check if directory exists
    if not os.path.exists(notes_directory):
        print(f"Error: Directory {notes_directory} does not exist")
        return 1
    
    # Load notes into DataFrame
    df = load_notes_to_dataframe(notes_directory)
    
    if df.empty:
        print("No notes found to process")
        return 1
    
    # Process annotations
    df_annotated = process_notes_dataframe(df)
    
    # Extract annotation results - CRITICAL: Use correct column names for evaluation
    # The evaluator expects 'time1' and 'date' columns (not 'FOLLOWUP_TIME' etc)
    
    # Normalize the annotations to match evaluation expectations
    def normalize_time(val):
        """Normalize time values - handle PRN and empty values"""
        if pd.isna(val) or str(val).strip() == '':
            return ''
        val = str(val).strip()
        if val.upper() == 'PRN':
            return 'PRN'
        return val
    
    def normalize_unit(val):
        """Normalize unit values - handle misc and empty values"""
        if pd.isna(val) or str(val).strip() == '':
            return ''
        val = str(val).strip().lower()
        if val == 'misc':
            return 'misc'
        # Normalize units to singular form
        if 'day' in val:
            return 'day'
        elif 'week' in val or 'wk' in val:
            return 'week'
        elif 'month' in val or 'mo' in val:
            return 'month'
        elif 'year' in val or 'yr' in val:
            return 'year'
        return val
    
    # Create the final DataFrame with correct column names
    output_df = pd.DataFrame()
    
    # Copy original columns
    output_df['STUDY_ID'] = df['STUDY_ID']
    output_df['CONTACT_DATE'] = df['CONTACT_DATE']
    output_df['NOTE_ID'] = df['NOTE_ID']
    output_df['NOTE_TEXT'] = df['NOTE_TEXT']
    output_df['FILENAME'] = df['FILENAME']
    output_df['NOTE_LENGTH'] = df['NOTE_LENGTH']
    
    # Add annotation columns with correct names
    output_df['time1'] = df_annotated['FOLLOWUP_TIME'].apply(normalize_time)
    output_df['date'] = df_annotated['FOLLOWUP_UNIT'].apply(normalize_unit)
    output_df['time2'] = df_annotated['FOLLOWUP_TIME2'].apply(normalize_time)
    output_df['hit'] = df_annotated['GLAUCOMA_HIT']
    output_df['hit word'] = df_annotated['HIT_WORD']
    
    # Handle PRN/misc cases - if unit is 'misc', set time to 'PRN'
    mask_misc = output_df['date'] == 'misc'
    output_df.loc[mask_misc, 'time1'] = 'PRN'
    
    # If no follow-up detected (empty unit), ensure time is also empty
    mask_no_fu = output_df['date'] == ''
    output_df.loc[mask_no_fu, 'time1'] = ''
    output_df.loc[mask_no_fu, 'time2'] = ''
    
    # Save to CSV
    output_df.to_csv(output_file_path, index=False)
    print(f"\nGlaucoma annotation results saved to: {output_file_path}")
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total notes processed: {len(output_df)}")
    print(f"Glaucoma hits: {sum(output_df['hit'] == 'Y')}")
    print(f"Follow-up times detected: {sum(output_df['time1'] != '')}")
    
    # Show glaucoma hit distribution
    glaucoma_counts = Counter(output_df['hit'])
    print(f"\nGlaucoma detection:")
    for key, value in glaucoma_counts.items():
        print(f"  {key}: {value} ({value/len(output_df)*100:.1f}%)")
    
    # Show follow-up unit distribution
    followup_counts = Counter([x for x in output_df['date'] if x != ''])
    if followup_counts:
        print(f"\nFollow-up intervals detected:")
        for key, value in followup_counts.items():
            print(f"  {key}: {value}")
    
    # Show sample results
    if args.verbose:
        print(f"\n=== SAMPLE RESULTS ===")
        sample_cols = ['STUDY_ID', 'CONTACT_DATE', 'hit', 'time1', 'date']
        print(output_df[sample_cols].head(10).to_string(index=False))
    
    print(f"\nComplete annotated dataset saved to: {output_file_path}")
    print("Ready for evaluation against gold standard!")
    
    return 0

if __name__ == "__main__":
    exit(main())