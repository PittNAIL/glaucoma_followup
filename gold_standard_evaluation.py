import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import warnings
import argparse
import os
import sys

warnings.filterwarnings('ignore')


def _robust_read_csv(path, expected_cols):
    """Read CSV handling common header/index issues.

    - Drops auto-generated index columns like 'Unnamed: 0' or leading '0'.
    - If expected columns aren't found, tries reading with header=1 (second row as header).
    - As a last resort, detects a header row by scanning for expected column names.
    """
    def sanitize(df):
        # Normalize column names
        df.columns = [str(c).strip() for c in df.columns]
        # Drop common index artifacts
        for bad in ('Unnamed: 0', ''):
            if bad in df.columns:
                df = df.drop(columns=[bad])
        # If the first column is a stray numeric index named '0', drop it
        if len(df.columns) > 0 and df.columns[0] == '0':
            df = df.drop(columns=['0'])
        df.columns = [str(c).strip() for c in df.columns]
        return df

    # 1) Try normal read
    df = pd.read_csv(path)
    df = sanitize(df)

    # Handle misformatted table where first row contains true headers across numeric columns
    def fix_numbered_header_frame(df0):
        cols = set(df0.columns)
        label_candidates = [c for c in df0.columns if 'label' in str(c).lower()]
        num_cols = [c for c in df0.columns if str(c).strip().isdigit()]
        if len(num_cols) >= 3:  # expect at least 0,1,2
            num_cols_sorted = sorted(num_cols, key=lambda x: int(str(x)))
            # Use first row as potential header row for these numeric columns
            if len(df0) > 0:
                head_vals = [str(v).strip() for v in df0.iloc[0][num_cols_sorted].tolist()]
                # If expected headers appear among these values, remap
                if set(expected_cols).issubset(set(head_vals)):
                    mapping = {}
                    for i, c in enumerate(num_cols_sorted):
                        if i < len(head_vals) and head_vals[i]:
                            mapping[c] = head_vals[i]
                    df1 = df0.rename(columns=mapping).copy()
                    # Drop the first row (header row that was stored as data)
                    df1 = df1.iloc[1:].reset_index(drop=True)
                    # Drop label-like column if present
                    for lab in label_candidates:
                        if lab in df1.columns:
                            try:
                                df1 = df1.drop(columns=[lab])
                            except Exception:
                                pass
                    return sanitize(df1)
        return df0

    if not set(expected_cols).issubset(df.columns):
        df = fix_numbered_header_frame(df)
        if set(expected_cols).issubset(df.columns):
            return df
    else:
        return df

    # 2) Try with header on the second row
    try:
        df2 = pd.read_csv(path, header=1)
        df2 = sanitize(df2)
        if set(expected_cols).issubset(df2.columns):
            return df2
    except Exception:
        pass

    # 3) Last resort: read without header, detect header row by scanning
    try:
        raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
        for i in range(min(10, len(raw))):
            row_vals = [str(v).strip() for v in raw.iloc[i].tolist()]
            if set(expected_cols).issubset(set(row_vals)):
                header = row_vals
                data = raw.iloc[i + 1 :].copy()
                data.columns = header
                data = sanitize(data)
                if set(expected_cols).issubset(data.columns):
                    return data
                break
    except Exception:
        pass

    raise ValueError(f"Expected columns {expected_cols} not found in {path}. Found: {list(pd.read_csv(path, nrows=0).columns)}")


def load_data(annotated_path, gold_standard_path):
    """Load both annotated and gold standard dataframes"""
    expected_core = ['STUDY_ID', 'NOTE_ID', 'FILENAME']

    # Use robust reader to handle malformed headers/index columns
    annotated_df = _robust_read_csv(annotated_path, expected_core)
    gold_df = _robust_read_csv(gold_standard_path, expected_core)

    # Harmonize annotation column names (time/unit/hit) across different sources
    def coerce_annotation_columns(df, source_label="annotated"):
        colnames = {c: c for c in df.columns}

        # If alternate names exist, map them into canonical names expected by evaluation
        def ensure_col(target, options, default=None):
            if target in df.columns:
                return
            for opt in options:
                if opt in df.columns:
                    df[target] = df[opt]
                    return
            if default is not None:
                df[target] = default

        # Follow-up numeric value
        ensure_col('time1', [
            'llm_time1', 'FOLLOWUP_TIME', 'followup_time', 'fu_time', 'time', 'TIME1'
        ], default=None)

        # Follow-up unit
        ensure_col('date', [
            'llm_date', 'FOLLOWUP_UNIT', 'followup_unit', 'fu_unit', 'unit', 'date_unit', 'DATE', 'UNIT'
        ], default=None)

        # Hit columns (glaucoma detection)
        ensure_col('hit', [
            'GLAUCOMA_HIT', 'DR_HIT', 'hit_pred', 'hit_gold'
        ])
        ensure_col('hit word', [
            'HIT_WORD', 'hit_word', 'HIT WORD'
        ])

        # Informative note if we had to synthesize any columns
        missing = [c for c in ['time1', 'date'] if c not in colnames]
        if any(c not in colnames for c in ['time1', 'date']):
            print(f"ℹ️  Normalized {source_label} columns; using canonical ['time1','date']")

        return df

    annotated_df = coerce_annotation_columns(annotated_df, source_label="annotated")
    gold_df = coerce_annotation_columns(gold_df, source_label="gold")

    # Normalize filenames for robust matching
    def normalize_filename(val):
        if pd.isna(val):
            return None
        s = str(val).strip().strip('"\'')
        s = s.replace('\\', '/')
        s = s.split('/')[-1]  # basename
        return s

    def filebase(val):
        v = normalize_filename(val)
        if not v:
            return None
        if '.' in v:
            return v.rsplit('.', 1)[0]
        return v

    for df in (annotated_df, gold_df):
        if 'FILENAME' in df.columns:
            df['FILENAME'] = df['FILENAME'].apply(normalize_filename)
            df['FILEBASE'] = df['FILENAME'].apply(filebase)
    
    # Check if columns have 'llm_' prefix (for LLM results)
    if 'llm_time1' in annotated_df.columns:
        # LLM results format - rename columns for consistency
        annotated_df = annotated_df.rename(columns={
            'llm_time1': 'time1',
            'llm_date': 'date'
        })
    
    # Ensure required annotation columns exist before selecting/merging
    for col in ['time1', 'date', 'hit']:
        if col not in annotated_df.columns:
            annotated_df[col] = None

    # For LLM results, we don't have 'hit' column, so we'll skip it
    gold_cols = ['STUDY_ID', 'NOTE_ID', 'FILENAME', 'time_goldstandard', 'date_goldstandard']
    if 'hit' in gold_df.columns:
        gold_cols.insert(3, 'hit')
    # Include NOTE_TEXT if available
    if 'NOTE_TEXT' in gold_df.columns:
        gold_cols.append('NOTE_TEXT')

    annotated_cols = ['STUDY_ID', 'NOTE_ID', 'FILENAME', 'time1', 'date']
    if 'hit' in annotated_df.columns:
        annotated_cols.insert(3, 'hit')

    # Guard against KeyError: if any columns still missing, create them and proceed
    for col in annotated_cols:
        if col not in annotated_df.columns:
            annotated_df[col] = None
    
    # Pre-process annotated data to normalize empty values
    for col in ['time1', 'date']:
        if col in annotated_df.columns:
            # Strip strings but preserve actual NaN
            annotated_df[col] = annotated_df[col].apply(lambda v: v.strip() if isinstance(v, str) else v)
            # Replace empty/invalid sentinel strings with NaN; keep lowercase 'none' as valid no-FU marker
            annotated_df[col] = annotated_df[col].replace({
                '': None, 'nan': None, 'NaN': None, 'null': None, 'Null': None, 'NONE': None, 'None': None
            })

    # Merge gold standard (300 notes) with annotated data
    merged_df = pd.merge(
        gold_df[gold_cols],
        annotated_df[annotated_cols],
        on=['FILENAME'],
        suffixes=('_gold', '_pred'),
        how='left'
    )

    # If poor match rate, try fallback on FILEBASE (filename without extension)
    matched = merged_df['NOTE_ID_pred'].notna().sum() if 'NOTE_ID_pred' in merged_df.columns else merged_df['time1'].notna().sum()
    if matched < max(1, int(0.5 * len(merged_df))):  # less than 50% matched
        if 'FILEBASE' in gold_df.columns and 'FILEBASE' in annotated_df.columns:
            gold_tmp = gold_df.copy()
            ann_tmp = annotated_df.copy()
            gold_tmp = gold_tmp.assign(FILEBASE=gold_tmp['FILENAME'].apply(filebase))
            ann_tmp = ann_tmp.assign(FILEBASE=ann_tmp['FILENAME'].apply(filebase))
            gold_cols_fb = gold_cols + (['FILEBASE'] if 'FILEBASE' not in gold_cols else [])
            ann_cols_fb = annotated_cols + (['FILEBASE'] if 'FILEBASE' not in annotated_cols else [])
            merged_fb = pd.merge(
                gold_tmp[gold_cols_fb],
                ann_tmp[ann_cols_fb],
                on=['FILEBASE'],
                suffixes=('_gold', '_pred'),
                how='left'
            )
            matched_fb = merged_fb['NOTE_ID_pred'].notna().sum() if 'NOTE_ID_pred' in merged_fb.columns else merged_fb['time1'].notna().sum()
            if matched_fb > matched:
                print(f"ℹ️  Using FILEBASE merge (improved matches: {matched_fb}/{len(merged_fb)} vs {matched}/{len(merged_df)})")
                merged_df = merged_fb

    # Diagnostics on matching
    total = len(merged_df)
    matched_rows = merged_df['time1'].notna().sum()
    print(f"🔗 Merge diagnostics: {matched_rows}/{total} records have predicted annotations after merge")
    
    # Check for any gold standard records without matches
    actually_missing = merged_df[merged_df['FILENAME'].isna()]
    if len(actually_missing) > 0:
        print(f"⚠️  Warning: {len(actually_missing)} gold standard records have no matching annotations")
    
    # Report on extraction coverage and empty annotations
    has_time_extraction = merged_df['time1'].notna().sum()
    has_date_extraction = merged_df['date'].notna().sum()
    empty_time_annotations = merged_df['time1'].isna().sum()
    empty_date_annotations = merged_df['date'].isna().sum()

    print(f"📊 Extraction coverage: {has_time_extraction}/{len(merged_df)} records have follow-up times extracted")
    print(f"📊 Date coverage: {has_date_extraction}/{len(merged_df)} records have date units extracted")
    print(f"📊 Empty annotations: {empty_time_annotations} time fields, {empty_date_annotations} date fields (treated as 'none')")
    
    return merged_df


def normalize_unit(unit):
    """Normalize unit strings to handle variations like 'months' vs 'month'
    Returns None for misc units, 'none', empty values (treating as no follow-up)"""
    if pd.isna(unit) or str(unit).strip() == '' or str(unit).strip().lower() == 'nan':
        return None

    unit = str(unit).lower().strip()

    # Treat misc, none, PRN as None (no follow-up)
    if unit == 'misc' or unit == 'none' or 'prn' in unit.lower() or unit == '':
        return None

    # Normalize to singular form
    if 'day' in unit:
        return 'day'
    elif 'week' in unit or 'wk' in unit:
        return 'week'
    elif 'month' in unit or 'mo' in unit:
        return 'month'
    elif 'year' in unit or 'yr' in unit:
        return 'year'
    else:
        return unit


def normalize_time_value(time):
    """Normalize time values, treating PRN, 'none', and empty values as None (no follow-up)"""
    if pd.isna(time) or str(time).strip() == '' or str(time).strip().lower() == 'nan':
        return None

    time_str = str(time).strip().upper()

    # Treat PRN and 'NONE' as None (no follow-up)
    if time_str == 'PRN' or time_str == 'NONE' or time_str == '':
        return None

    try:
        return float(time_str)
    except:
        return None


def convert_to_days(time, unit):
    """Convert time to days for comparison
    Returns None for PRN/misc (treating as no follow-up)"""
    if time is None or unit is None:
        return None
    
    try:
        time = float(time)
    except:
        return None
    
    if unit == 'day':
        return time
    elif unit == 'week':
        return time * 7
    elif unit == 'month':
        return time * 30.44
    elif unit == 'year':
        return time * 365.25
    else:
        return None


def evaluate_followup_extraction_detailed(df):
    """Evaluate follow-up time extraction with detailed analysis"""
    print("="*60)
    print("FOLLOW-UP TIME EXTRACTION PERFORMANCE (DETAILED)")
    print("="*60)
    print("\n📝 NOTE: 'none', PRN, misc, and empty annotations are treated as 'no follow-up' per gold standard")
    
    # Normalize units and times (PRN and misc become None)
    df['pred_time_norm'] = df['time1'].apply(normalize_time_value)
    df['gold_time_norm'] = df['time_goldstandard'].apply(normalize_time_value)
    df['pred_unit_norm'] = df['date'].apply(normalize_unit)
    df['gold_unit_norm'] = df['date_goldstandard'].apply(normalize_unit)
    
    # Convert to days for comparison
    df['pred_days'] = df.apply(lambda x: convert_to_days(x['pred_time_norm'], x['pred_unit_norm']), axis=1)
    df['gold_days'] = df.apply(lambda x: convert_to_days(x['gold_time_norm'], x['gold_unit_norm']), axis=1)
    
    # Identify which records have follow-up times (PRN/misc are now None)
    df['has_pred_fu'] = df['pred_days'].notna()
    df['has_gold_fu'] = df['gold_days'].notna()
    
    # Calculate detection metrics
    precision = precision_score(df['has_gold_fu'], df['has_pred_fu'])
    recall = recall_score(df['has_gold_fu'], df['has_pred_fu'])
    f1 = f1_score(df['has_gold_fu'], df['has_pred_fu'])
    accuracy = accuracy_score(df['has_gold_fu'], df['has_pred_fu'])
    
    print("\n📊 FOLLOW-UP DETECTION METRICS (Binary - Found vs Not Found):")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1 Score: {f1:.3f}")
    print(f"  Accuracy: {accuracy:.3f}")
    
    # Confusion matrix for detection
    cm = confusion_matrix(df['has_gold_fu'], df['has_pred_fu'])
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n📊 CONFUSION MATRIX:")
    print(f"  True Positives (correctly found FU): {tp}")
    print(f"  True Negatives (correctly no FU): {tn}")
    print(f"  False Positives (wrongly found FU): {fp}")
    print(f"  False Negatives (missed FU): {fn}")
    
    # Show PRN/misc/empty/none handling statistics
    prn_misc_pred = df['time1'].astype(str).str.upper().eq('PRN') | df['date'].astype(str).str.lower().eq('misc')
    prn_misc_gold = df['time_goldstandard'].astype(str).str.upper().eq('PRN') | df['date_goldstandard'].astype(str).str.lower().eq('misc')

    none_pred = df['time1'].astype(str).str.lower().eq('none') | df['date'].astype(str).str.lower().eq('none')
    none_gold = df['time_goldstandard'].astype(str).str.lower().eq('none') | df['date_goldstandard'].astype(str).str.lower().eq('none')

    empty_pred = df['time1'].isna() | df['date'].isna()
    empty_gold = df['time_goldstandard'].isna() | df['date_goldstandard'].isna()

    print(f"\n📊 NO FOLLOW-UP ANNOTATION BREAKDOWN:")
    print(f"  Predicted:")
    print(f"    - 'none' annotations (valid no-FU): {none_pred.sum()}")
    print(f"    - PRN/misc cases (treated as no FU): {prn_misc_pred.sum()}")
    print(f"    - Empty/missing annotations: {empty_pred.sum()}")
    print(f"  Gold standard:")
    print(f"    - 'none' annotations (valid no-FU): {none_gold.sum()}")
    print(f"    - PRN/misc cases (treated as no FU): {prn_misc_gold.sum()}")
    print(f"    - Empty/missing annotations: {empty_gold.sum()}")

    # Calculate correct 'none' matches
    both_none = none_pred & none_gold
    print(f"  Correctly identified 'none' cases: {both_none.sum()}")
    
    # For records where both have follow-up times, check accuracy
    both_have_fu = df[df['has_pred_fu'] & df['has_gold_fu']].copy()
    
    exact_accuracy = 0
    days_accuracy = 0
    mae_days = 0
    
    if len(both_have_fu) > 0:
        print(f"\n📊 VALUE ACCURACY (When Both Detected Follow-up):")
        print(f"  Total cases where both detected FU: {len(both_have_fu)}")
        
        # Check exact time and unit match (after normalization)
        both_have_fu['time_match'] = both_have_fu['pred_time_norm'] == both_have_fu['gold_time_norm']
        both_have_fu['unit_match'] = both_have_fu['pred_unit_norm'] == both_have_fu['gold_unit_norm']
        both_have_fu['exact_match'] = both_have_fu['time_match'] & both_have_fu['unit_match']
        
        # Check if values are equivalent in days (within 1 day tolerance)
        both_have_fu['days_match'] = np.abs(both_have_fu['pred_days'] - both_have_fu['gold_days']) < 1
        
        # Calculate different accuracy levels
        time_accuracy = both_have_fu['time_match'].mean()
        unit_accuracy = both_have_fu['unit_match'].mean()
        exact_accuracy = both_have_fu['exact_match'].mean()
        days_accuracy = both_have_fu['days_match'].mean()
        
        print(f"  Time value match rate: {time_accuracy:.3f} ({both_have_fu['time_match'].sum()}/{len(both_have_fu)})")
        print(f"  Unit match rate (normalized): {unit_accuracy:.3f} ({both_have_fu['unit_match'].sum()}/{len(both_have_fu)})")
        print(f"  Exact match rate (time & unit): {exact_accuracy:.3f} ({both_have_fu['exact_match'].sum()}/{len(both_have_fu)})")
        print(f"  Days equivalent match rate: {days_accuracy:.3f} ({both_have_fu['days_match'].sum()}/{len(both_have_fu)})")
        
        # Mean absolute error
        mae_days = np.abs(both_have_fu['pred_days'] - both_have_fu['gold_days']).mean()
        print(f"  Mean Absolute Error: {mae_days:.1f} days")
        
        # Show mismatches for debugging
        mismatches = both_have_fu[~both_have_fu['days_match']]
        if len(mismatches) > 0:
            print(f"\n📊 MISMATCHES (first 10):")
            for idx, row in mismatches.head(10).iterrows():
                print(f"  File: {row['FILENAME']}")
                print(f"    Predicted: {row['time1']} {row['date']} → {row['pred_time_norm']} {row['pred_unit_norm']} ({row['pred_days']:.0f} days)")
                print(f"    Gold: {row['time_goldstandard']} {row['date_goldstandard']} → {row['gold_time_norm']} {row['gold_unit_norm']} ({row['gold_days']:.0f} days)")
                print()
    
    # Analyze false positives
    if fp > 0:
        print(f"\n📊 FALSE POSITIVES (found FU when shouldn't - first 5):")
        false_positives = df[df['has_pred_fu'] & ~df['has_gold_fu']]
        for idx, row in false_positives.head(5).iterrows():
            print(f"  File: {row['FILENAME']}")
            print(f"    Predicted: {row['time1']} {row['date']}")
            if pd.notna(row['time_goldstandard']) or pd.notna(row['date_goldstandard']):
                print(f"    Gold had: {row['time_goldstandard']} {row['date_goldstandard']} (treated as no FU)")
    
    # Analyze false negatives
    if fn > 0:
        print(f"\n📊 FALSE NEGATIVES (missed FU - first 5):")
        false_negatives = df[~df['has_pred_fu'] & df['has_gold_fu']]
        for idx, row in false_negatives.head(5).iterrows():
            print(f"  File: {row['FILENAME']}")
            print(f"    Gold: {row['time_goldstandard']} {row['date_goldstandard']}")
            if pd.notna(row['time1']) or pd.notna(row['date']):
                print(f"    Predicted had: {row['time1']} {row['date']} (treated as no FU)")
    
    return {
        'detection_precision': precision,
        'detection_recall': recall,
        'detection_f1': f1,
        'detection_accuracy': accuracy,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'exact_match_rate': exact_accuracy,
        'days_match_rate': days_accuracy,
        'mae_days': mae_days
    }


def evaluate_glaucoma_detection(df):
    """Evaluate glaucoma hit detection performance"""
    # Check if hit columns exist
    if 'hit_pred' not in df.columns or 'hit_gold' not in df.columns:
        print("\n" + "="*60)
        print("GLAUCOMA DETECTION PERFORMANCE")
        print("="*60)
        print("  ⚠️  Hit detection columns not found - skipping glaucoma detection evaluation")
        return None
    
    print("\n" + "="*60)
    print("GLAUCOMA DETECTION PERFORMANCE")
    print("="*60)
    
    # Clean and prepare the data
    df['hit_pred_binary'] = df['hit_pred'].apply(lambda x: 1 if str(x).upper() == 'Y' else 0)
    df['hit_gold_binary'] = df['hit_gold'].apply(lambda x: 1 if str(x).upper() == 'Y' else 0)
    
    # Calculate metrics
    precision = precision_score(df['hit_gold_binary'], df['hit_pred_binary'])
    recall = recall_score(df['hit_gold_binary'], df['hit_pred_binary'])
    f1 = f1_score(df['hit_gold_binary'], df['hit_pred_binary'])
    accuracy = accuracy_score(df['hit_gold_binary'], df['hit_pred_binary'])
    
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1 Score: {f1:.3f}")
    print(f"  Accuracy: {accuracy:.3f}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }


def create_performance_summary(followup_metrics, glaucoma_metrics=None):
    """Create a comprehensive performance summary"""
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    print("\n🎯 PRIMARY METRIC - FOLLOW-UP TIME EXTRACTION:")
    print(f"  Detection F1 Score: {followup_metrics['detection_f1']:.3f}")
    print(f"  Detection Precision: {followup_metrics['detection_precision']:.3f}")
    print(f"  Detection Recall: {followup_metrics['detection_recall']:.3f}")
    print(f"  Value Accuracy (when both detected): {followup_metrics['days_match_rate']:.3f}")
    
    if glaucoma_metrics:
        print(f"\n🎯 SECONDARY METRIC - GLAUCOMA DETECTION:")
        print(f"  F1 Score: {glaucoma_metrics['f1']:.3f}")
        print(f"  Precision: {glaucoma_metrics['precision']:.3f}")
        print(f"  Recall: {glaucoma_metrics['recall']:.3f}")
    
    # Overall assessment based on follow-up extraction
    fu_f1 = followup_metrics['detection_f1']
    fu_value_acc = followup_metrics['days_match_rate']
    combined_score = (fu_f1 + fu_value_acc) / 2
    
    print(f"\n📈 OVERALL ASSESSMENT:")
    print(f"  Combined Follow-up Score: {combined_score:.3f}")
    
    if combined_score >= 0.9:
        interpretation = "Excellent Performance"
    elif combined_score >= 0.8:
        interpretation = "Good Performance"
    elif combined_score >= 0.7:
        interpretation = "Satisfactory Performance"
    elif combined_score >= 0.6:
        interpretation = "Moderate Performance"
    else:
        interpretation = "Needs Improvement"
    
    print(f"  Interpretation: {interpretation}")
    
    # Provide specific feedback
    print(f"\n💡 KEY INSIGHTS:")
    if followup_metrics['detection_recall'] < 0.8:
        print(f"  - Consider improving recall ({followup_metrics['detection_recall']:.3f}) - missing some follow-ups")
    if followup_metrics['detection_precision'] < 0.8:
        print(f"  - Consider improving precision ({followup_metrics['detection_precision']:.3f}) - too many false positives")
    if followup_metrics['days_match_rate'] < 0.8:
        print(f"  - Value extraction accuracy could be improved ({followup_metrics['days_match_rate']:.3f})")
    if followup_metrics['mae_days'] > 30:
        print(f"  - High error in time values (MAE: {followup_metrics['mae_days']:.1f} days)")


def setup_argparse():
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description='Evaluate annotation results against gold standard',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with required files:
  python evaluate.py -a annotations.csv -g gold_standard.csv

  # Specify custom output file:
  python evaluate.py -a annotations.csv -g gold_standard.csv -o results.csv

  # Interactive mode (prompts for file paths):
  python evaluate.py --interactive
  
  # Verbose mode for extra debugging info:
  python evaluate.py -a annotations.csv -g gold_standard.csv --verbose
        """
    )
    
    parser.add_argument('-a', '--annotated', 
                        type=str,
                        help='Path to the annotated results CSV file')
    
    parser.add_argument('-g', '--gold', 
                        type=str,
                        help='Path to the gold standard CSV file')
    
    parser.add_argument('-o', '--output', 
                        type=str,
                        help='Path for output results CSV (default: evaluation_results.csv in current directory)')
    
    parser.add_argument('-i', '--interactive',
                        action='store_true',
                        help='Interactive mode - prompts for file paths')
    
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Verbose output for debugging')
    
    return parser


def get_file_path_interactive(prompt, check_exists=True):
    """Interactive prompt for file path with validation"""
    while True:
        path = input(prompt).strip()
        
        # Handle quotes around path
        if path.startswith('"') and path.endswith('"'):
            path = path[1:-1]
        if path.startswith("'") and path.endswith("'"):
            path = path[1:-1]
        
        # Expand user path
        path = os.path.expanduser(path)
        
        if check_exists:
            if os.path.exists(path):
                return path
            else:
                print(f"❌ File not found: {path}")
                retry = input("Try again? (y/n): ").lower()
                if retry != 'y':
                    return None
        else:
            # For output file, just check if directory exists
            directory = os.path.dirname(path)
            if directory == '' or os.path.exists(directory):
                return path
            else:
                print(f"❌ Directory does not exist: {directory}")
                retry = input("Try again? (y/n): ").lower()
                if retry != 'y':
                    return None


def main():
    """Main evaluation function with flexible input options"""
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Determine how to get file paths
    if args.interactive or (not args.annotated and not args.gold):
        # Interactive mode
        print("🔍 ANNOTATION EVALUATION TOOL - Interactive Mode")
        print("="*60)
        
        annotated_path = get_file_path_interactive(
            "📁 Enter path to annotated results CSV: "
        )
        if not annotated_path:
            print("Evaluation cancelled.")
            return
        
        gold_standard_path = get_file_path_interactive(
            "📁 Enter path to gold standard CSV: "
        )
        if not gold_standard_path:
            print("Evaluation cancelled.")
            return
        
        output_path = get_file_path_interactive(
            "📁 Enter output file path (or press Enter for default): ",
            check_exists=False
        )
        if not output_path or output_path == '':
            output_path = "evaluation_results.csv"
    
    else:
        # Command line arguments mode
        if not args.annotated or not args.gold:
            parser.error("Both --annotated and --gold are required unless using --interactive")
        
        annotated_path = os.path.expanduser(args.annotated)
        gold_standard_path = os.path.expanduser(args.gold)
        output_path = args.output if args.output else "evaluation_results.csv"
        
        # Validate input files exist
        if not os.path.exists(annotated_path):
            print(f"❌ Error: Annotated file not found: {annotated_path}")
            sys.exit(1)
        
        if not os.path.exists(gold_standard_path):
            print(f"❌ Error: Gold standard file not found: {gold_standard_path}")
            sys.exit(1)
    
    # Show file paths being used
    print("\n📂 File paths:")
    print(f"  Annotated results: {annotated_path}")
    print(f"  Gold standard: {gold_standard_path}")
    print(f"  Output will be saved to: {output_path}")
    print()
    
    # Load and evaluate data
    print("🔍 Loading data...")
    try:
        merged_df = load_data(annotated_path, gold_standard_path)
        print(f"✅ Successfully loaded and merged {len(merged_df)} records\n")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        print("\nPlease check that both CSV files exist and have the correct columns.")
        print("Expected columns in both files: STUDY_ID, NOTE_ID, FILENAME")
        print("Expected columns in annotated: time1 (or llm_time1), date (or llm_date)")
        print("Expected columns in gold standard: time_goldstandard, date_goldstandard")
        sys.exit(1)
    
    # Primary evaluation: Follow-up extraction
    try:
        followup_metrics = evaluate_followup_extraction_detailed(merged_df)
    except Exception as e:
        print(f"❌ Error during follow-up evaluation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    # Secondary evaluation: Glaucoma detection (optional)
    try:
        glaucoma_metrics = evaluate_glaucoma_detection(merged_df)
    except Exception as e:
        print(f"⚠️  Warning during glaucoma evaluation: {e}")
        glaucoma_metrics = None
        if args.verbose:
            import traceback
            traceback.print_exc()
    
    # Create comprehensive summary
    create_performance_summary(followup_metrics, glaucoma_metrics)
    
    # Save detailed results
    try:
        # Add evaluation columns to the dataframe
        merged_df['pred_time_norm'] = merged_df['time1'].apply(normalize_time_value)
        merged_df['gold_time_norm'] = merged_df['time_goldstandard'].apply(normalize_time_value)
        merged_df['pred_unit_norm'] = merged_df['date'].apply(normalize_unit)
        merged_df['gold_unit_norm'] = merged_df['date_goldstandard'].apply(normalize_unit)

        # Convert to days for better comparison across units
        merged_df['pred_days'] = merged_df.apply(lambda x: convert_to_days(x['pred_time_norm'], x['pred_unit_norm']), axis=1)
        merged_df['gold_days'] = merged_df.apply(lambda x: convert_to_days(x['gold_time_norm'], x['gold_unit_norm']), axis=1)

        # Match logic
        def check_match(row):
            if pd.notna(row['pred_days']) and pd.notna(row['gold_days']):
                return abs(row['pred_days'] - row['gold_days']) < 1.0
            elif pd.isna(row['pred_days']) and pd.isna(row['gold_days']):
                return True
            else:
                return False

        merged_df['match'] = merged_df.apply(check_match, axis=1)

        # Save with all the calculated columns
        merged_df.to_csv(output_path, index=False)
        print(f"\n✅ Detailed evaluation results saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    return merged_df, followup_metrics, glaucoma_metrics


if __name__ == "__main__":
    results = main()