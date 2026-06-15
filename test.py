

from General.dataset_sample import split_multiple_train_test


train_list, test_list = split_multiple_train_test(
    [
        "/data2/xiangcen/data/pet_gen/processed/batch1_h5_v2",
        "/data2/xiangcen/data/pet_gen/processed/batch2_h5_v2",
        "/data2/xiangcen/data/pet_gen/processed/batch3_h5_v2",
    ],
    [40, 40, 20],
)


print(test_list[:3])




train_list, test_list = split_multiple_train_test(
    [
        "/data2/xiangcen/data/pet_gen/processed/warped_fdg_pet_h5/batch1_h5_v2",
        "/data2/xiangcen/data/pet_gen/processed/warped_fdg_pet_h5/batch2_h5_v2",
        "/data2/xiangcen/data/pet_gen/processed/warped_fdg_pet_h5/batch3_h5_v2",
    ],
    [40, 40, 20],
)


print(test_list[:3])