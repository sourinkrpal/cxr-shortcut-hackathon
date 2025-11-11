import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

root_dir = Path('/deep_learning/output/petersen/cxr_small_data')

# normal cxp train set data
train_df = pd.read_csv(root_dir / 'CheXpert-v1.0-small/train.csv').set_index('Path')

# add the soft drain labels determined by our drain classifier
train_drain_df = pd.read_csv(root_dir / 'train_drain.csv').set_index('path')
train_df["Drain"] = pd.NA
train_df.update({'Drain': train_drain_df.soft_drain})
print(train_df.Drain.head())

# add / overwrite with manual drain labels
drain_data = pd.read_csv(root_dir / 'drain_splits.csv')
drain_data.Path = drain_data.Path.str.replace('CheXpert-v1.0/', 'CheXpert-v1.0-small/', regex=False)
drain_data = drain_data.set_index('Path')
train_df.update({'Drain': drain_data.Drain})
print(train_df.Drain.head())

# now all the same things but for the val set
val_df = pd.read_csv(root_dir / 'CheXpert-v1.0-small/valid.csv').set_index('Path')

# add the soft drain labels determined by our drain classifier
val_drain_df = pd.read_csv(root_dir / 'valid_drain.csv').set_index('path')
val_df["Drain"] = pd.NA
val_df.update({"Drain": val_drain_df.soft_drain})
print(val_df.Drain.head())

# add / overwrite with manual drain labels
# none of the manual annotations are from the val set, apparently
val_df.update({"Drain": drain_data.Drain})

# Now merge the two, yielding a final full dataset with drain labels
full_df = pd.concat([train_df, val_df], ignore_index=False)
print(full_df.Drain.head())
assert pd.isna(full_df.Drain).sum() > 5000
print(f'{len(full_df)} samples initially')

# Drop lateral images
full_df = full_df[full_df['Frontal/Lateral'] == 'Frontal']
print(f'{len(full_df)} samples after dropping laterals')

# Drop uncertain Pneumothorax cases
full_df = full_df[~(full_df.Pneumothorax == -1)]
full_df = full_df[~pd.isna(full_df.Pneumothorax)]
print(f'{len(full_df)} samples after dropping uncertain Pneumothorax cases')

# Drop uncertain drain cases
full_df = full_df[~pd.isna(full_df.Drain)]
print(f'{len(full_df)} samples after dropping uncertain drain cases')

# Use only one image per patient
# Prefer with Pneumothorax if exists
# Prefer with drain label if exists
def select_best_record_per_patient(df):
    """
    Select one record per patient with prioritization:
    - Prefer Pneumothorax=1 cases
    - For Pneumothorax=1: prefer Drain=0
    - For Pneumothorax=0: prefer Drain=1
    """
    # Extract patient ID from Path
    df = df.copy()
    df['Path'] = df.index
    df['patient_id'] = df.Path.str.extract(r'patient(\d+)')[0]
    
    # Create priority score (higher is better)
    # Priority 1: Pneumothorax=1, Drain=0 (score=4)
    # Priority 2: Pneumothorax=1, Drain=1 (score=3)
    # Priority 3: Pneumothorax=0, Drain=1 (score=2)
    # Priority 4: Pneumothorax=0, Drain=0 (score=1)
    df['priority'] = (
        (df['Pneumothorax'] == 1) * 2 +  # Pneumothorax=1 adds 2 points
        ((df['Pneumothorax'] == 1) & (df['Drain'] == 0)) * 1 +  # Ptx=1 & Drain=0 adds 1 more
        ((df['Pneumothorax'] == 0) & (df['Drain'] == 1)) * 1    # Ptx=0 & Drain=1 adds 1
    )
    
    # Sort by patient_id and priority (descending), then keep first record per patient
    df_sorted = df.sort_values(['patient_id', 'priority'], ascending=[True, False])
    df_result = df_sorted.groupby('patient_id', as_index=False).first()
    
    # Set Path back as index
    df_result = df_result.set_index('Path') 
    
    # Drop helper columns
    df_result = df_result.drop(columns=['patient_id', 'priority'])
    
   # Extract patient IDs again for assertion
    result_patient_ids = df_result.index.str.extract(r'patient(\d+)')[0]
    
    # Assert all patient IDs are unique
    assert result_patient_ids.nunique() == len(df_result), \
        f"Duplicate patient IDs found! Unique: {result_patient_ids.nunique()}, Total: {len(df_result)}"
    
    return df_result

full_df = select_best_record_per_patient(full_df)
print(f'{len(full_df)} samples after selecting a single recording per patient')

# Summary stats
pneu_msk = full_df.Pneumothorax == 1
drain_msk = full_df.Drain == 1
print(f"{(pneu_msk & drain_msk).sum()} cases with pneu + drain")
print(f"{(pneu_msk & ~drain_msk).sum()} cases with pneu + no drain")
print(f"{(~pneu_msk & drain_msk).sum()} cases with no pneu + drain")
print(f"{(~pneu_msk & ~drain_msk).sum()} cases with no pneu + no drain")

# Train/val/test split
# Train+val: 80% of pneus have drains; 80% of no-pneus have no drains
# Test-aligned: same
# Test-misaligned: opposite
test_aligned = pd.concat([full_df[pneu_msk & drain_msk].sample(20), full_df[pneu_msk & ~drain_msk].sample(5),
                          full_df[~pneu_msk & ~drain_msk].sample(20), full_df[~pneu_msk & drain_msk].sample(5)], 
                         ignore_index=False)
used = full_df.index.isin(test_aligned.index)  # index is Path

test_misaligned = pd.concat([full_df[~used & pneu_msk & ~drain_msk].sample(20), full_df[~used & pneu_msk & drain_msk].sample(5),
                          full_df[~used & ~pneu_msk & drain_msk].sample(20), full_df[~used & ~pneu_msk & ~drain_msk].sample(5)], 
                         ignore_index=False)                         
used = used | (full_df.index.isin(test_misaligned.index))  # index is Path

# VERSION 1: Train + val identically sampled
# critical group: pneu + drain
n_base = (pneu_msk & drain_msk & ~used).sum()

# use this as basis + calc remaining group sizes from this
train_val = pd.concat([full_df[~used & pneu_msk & drain_msk].sample(n_base), full_df[~used & pneu_msk & ~drain_msk].sample(int(n_base/3)),
                       full_df[~used & ~pneu_msk & drain_msk].sample(int(n_base/3)), full_df[~used & ~pneu_msk & ~drain_msk].sample(n_base)],
                      ignore_index=False)
train_val['strat_col'] = train_val.Pneumothorax.astype(str) + '_' + train_val.Drain.astype(str)
train, val = train_test_split(train_val, test_size=0.2, stratify=train_val.strat_col, random_state=42)

train.to_csv(root_dir / 'train_drain_shortcut.csv')
val.to_csv(root_dir / 'val_drain_shortcut.csv')
test_aligned.to_csv(root_dir / 'test_drain_shortcut_aligned.csv')
test_misaligned.to_csv(root_dir / 'test_drain_shortcut_misaligned.csv')

# VERSION 2: balanced val set
val = pd.concat([full_df[pneu_msk & drain_msk].sample(25), full_df[pneu_msk & ~drain_msk].sample(25),
                          full_df[~pneu_msk & ~drain_msk].sample(25), full_df[~pneu_msk & drain_msk].sample(25)], 
                         ignore_index=False)                  
used = used | (full_df.index.isin(val.index))  # index is Path

# critical group: pneu + drain
n_base = (pneu_msk & drain_msk & ~used).sum()

# use this as basis + calc remaining group sizes from this
train = pd.concat([full_df[~used & pneu_msk & drain_msk].sample(n_base), full_df[~used & pneu_msk & ~drain_msk].sample(int(n_base/3)),
                       full_df[~used & ~pneu_msk & drain_msk].sample(int(n_base/3)), full_df[~used & ~pneu_msk & ~drain_msk].sample(n_base)],
                      ignore_index=False)
train.to_csv(root_dir / 'train_drain_shortcut_v2.csv')
val.to_csv(root_dir / 'val_drain_shortcut_v2.csv')