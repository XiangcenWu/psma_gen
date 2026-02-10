import os
import random

def split_train_test(patient_dir, num_validation=20, seed=325):

    patient_list = [os.path.join(patient_dir, f) for f in os.listdir(patient_dir) if f.endswith('.h5')]


    patient_list.sort()


    random.seed(seed)
    random.shuffle(patient_list)


    val_list = patient_list[:num_validation]
    train_list = patient_list[num_validation:]

    print(f"Total: {len(patient_list)} | Train: {len(train_list)} | Val: {len(val_list)}")
    
    return train_list, val_list

def split_multiple_train_test(dirs, num_validations, seed=325):
    if len(dirs) != len(num_validations):
        raise ValueError("dirs and num_validations must have the same length")
    all_train = []
    all_val = []

    for idx, patient_dir in enumerate(dirs):
        _dir = dirs[idx]
        num_val = num_validations[idx]

        train, val = split_train_test(_dir, num_val, seed)

        all_train.append(train)
        all_val.append(val)





    print(f"Train: {len(all_train)} | Val: {len(all_val)}")

    return all_train, all_val


