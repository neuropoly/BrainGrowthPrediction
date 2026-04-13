import os
import argparse

import pandas as pd
import nibabel as nib
from tqdm import tqdm
from bgp import const


def make_csv_A(df):
    """
    Creates CSV A, which adds the normalized region volumes in the existing CSV.
    The regional volumes are measured from SynthSeg segmentation and normalized
    using the training samples as a reference. 
    """ 
    coarse_regions  = const.COARSE_REGIONS
    code_map        = const.SYNTHSEG_CODEMAP

    records = []
    for record in tqdm(df.to_dict(orient='records')):
        segm = nib.load(record['segm_path']).get_fdata().round()
        record['head_size'] = (segm > 0).sum()
        
        for region in coarse_regions: 
            record[region] = 0
        
        for code, region in code_map.items():
            if region == 'background': continue
            coarse_region = region.replace('left_', '').replace('right_', '')
            record[coarse_region] += (segm == code).sum()
        
        records.append(record)
    
    csv_a_df = pd.DataFrame(records)
    
    for region in coarse_regions:
        # normalize volumes using min-max scaling
        # train_values = csv_a_df[ csv_a_df.split == 'train' ][region]
        train_values = csv_a_df[csv_a_df["split"].isin([1, 2, 3, 4])][region]
        minv, maxv = train_values.min(), train_values.max()
        print(f'For Region: {region}')
        print(f'min: {minv}, max: {maxv}')
        print(f'mean: {train_values.mean()}, std: {train_values.std()}')
        print(f'median: {train_values.median()}')
        csv_a_df[region] = (csv_a_df[region] - minv) / (maxv - minv)
    
    # print('FOR CSV A: \n')
    # for region in coarse_regions:
    #     # Extract training values for the region
    #     train_values = csv_a_df[csv_a_df.split == 'train'][region]

    #     # Compute statistics
    #     minv = train_values.min()
    #     maxv = train_values.max()
    #     meanv = train_values.mean()
    #     stdv = train_values.std()
    #     medianv = train_values.median()

    #     # Print region stats
    #     print(f"For Region: {region}")
    #     print(f"  min:    {minv:.3f}")
    #     print(f"  max:    {maxv:.3f}")
    #     print(f"  mean:   {meanv:.3f}")
    #     print(f"  std:    {stdv:.3f}")
    #     print(f"  median: {medianv:.3f}")

    return csv_a_df


def make_csv_B(df):
    """
    Creates CSV B, which contains all possible pairs (x_a, x_b) such that 
    both scans belong to the same patient and scan x_a is acquired before scan x_b.
    """
    sorting_field = 'months_to_screening' if 'months_to_screening' in df.columns else 'age'

    data = []
    for subject_id in tqdm(df.subject_id.unique()):
        subject_df = df[ df.subject_id == subject_id ].sort_values(sorting_field, ascending=True)
        for i in range(len(subject_df)):
            for j in range(i+1, len(subject_df)):
                s_rec = subject_df.iloc[i]
                e_rec = subject_df.iloc[j]
                record = { 'subject_id': s_rec.subject_id, 'sex': s_rec.sex, 'split': s_rec.split }
                remaining_columns = set(df.columns).difference(set(record.keys()))
                for column in remaining_columns:
                    record[f'starting_{column}'] = s_rec[column]
                    record[f'followup_{column}'] = e_rec[column]
                data.append(record)
    return pd.DataFrame(data)

def make_csv_trios(df):
    """
    Creates CSV of scan trios: for each subject, all combinations (x_a, x_b, x_c)
    such that x_a < x_b < x_c in acquisition order.
    """
    coarse_regions  = const.COARSE_REGIONS
    code_map        = const.SYNTHSEG_CODEMAP

    records = []
    for record in tqdm(df.to_dict(orient='records')):
        segm = nib.load(record['segm_path']).get_fdata().round()
        record['head_size'] = (segm > 0).sum()
        
        for region in coarse_regions: 
            record[region] = 0
        
        for code, region in code_map.items():
            if region == 'background': continue
            coarse_region = region.replace('left_', '').replace('right_', '')
            record[coarse_region] += (segm == code).sum()
        
        records.append(record)
    
    df = pd.DataFrame(records)

    # decide which field to sort on
    sorting_field = 'months_to_screening' if 'months_to_screening' in df.columns else 'age'

    triplets = []
    for subject_id in tqdm(df.subject_id.unique(), desc="Subjects"):
        subj = df[df.subject_id == subject_id].sort_values(sorting_field, ascending=True)
        n = len(subj)
        # for each ordered triple i < j < k
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    rec_a = subj.iloc[i]
                    rec_b = subj.iloc[j]
                    rec_c = subj.iloc[k]

                    # start with common keys
                    base = {
                        'subject_id': rec_a.subject_id,
                        'sex':         rec_a.sex,
                        'split':       rec_a.split
                    }

                    # determine which columns remain to expand
                    other_cols = [c for c in df.columns if c not in base]

                    # build the triple record
                    record = base.copy()
                    for col in other_cols:
                        if col == 'age':
                            record[f'starting_{col}']   = ((rec_a[col] * 6) - 1.0) / (6.9 - 1.0)
                            record[f'followup1_{col}']  = ((rec_b[col] * 6) - 1.0) / (6.9 - 1.0)
                            record[f'followup2_{col}']  = ((rec_c[col] * 6) - 1.0) / (6.9 - 1.0)
                        else:
                            record[f'starting_{col}']   = rec_a[col]
                            record[f'followup1_{col}']  = rec_b[col]
                            record[f'followup2_{col}']  = rec_c[col]


                    triplets.append(record)

    return pd.DataFrame(triplets)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv',      type=str, required=True)
    parser.add_argument('--output_path',      type=str, required=True)
    # parser.add_argument('--coarse_regions',   type=str, required=True)
    
    args = parser.parse_args()
    print(f'Reading dataset: {args.dataset_csv}')
    # read the dataset
    df = pd.read_csv(args.dataset_csv)

    # print()
    print('> Creating CSV A\n')
    csv_A = make_csv_A(df)
    csv_A.to_csv(os.path.join(args.output_path, 'A_datasets_complete.csv'), index=False)
    # if "KOALA" in args.dataset_csv:
    #     csv_A.to_csv(os.path.join(args.output_path, 'A_koala.csv'), index=False)
    # else:
    #     csv_A.to_csv(os.path.join(args.output_path, 'A.csv'), index=False)

    # print()
    # print('> Creating CSV B\n')
    # csv_B = make_csv_B(csv_A)
    # csv_B.to_csv(os.path.join(args.output_path, 'B.csv'), index=False)

    # print()
    # print('> Creating CSV C\n')
    # csv_C = make_csv_trios(df) # reading csv A
    # csv_C.to_csv(os.path.join(args.output_path, 'C_nonnormed_vol.csv'), index=False)

    print('> Creating CSV B\n')
    csv_B = make_csv_B(csv_A)
    csv_B.to_csv(os.path.join(args.output_path, 'B_datasets_complete.csv'), index=False)

    # CSV for external validation
    # subset = df.loc[df['dataset'] == 'hc-new-england']
    # csv_B = make_csv_B(subset)
    # csv_B.to_csv(os.path.join(args.output_path, 'external_long.csv'), index=False)
