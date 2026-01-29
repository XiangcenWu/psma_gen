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

