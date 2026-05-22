
import sys
import os
import argparse
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from monai.networks.nets import SwinUNETR
from General.data_loader import create_data_loader, ReadH5d
import torch.nn as nn

from General.dataset_sample import split_multiple_train_test
from Registration.training import get_registration_input_keys, make_identity_grid_m11

from inferencing import inference_batch


DEFAULT_RESULT_DIR = "/share/home/xcwu/pet_reg_results"


def make_output_path(weights_path, result_dir=DEFAULT_RESULT_DIR):
    os.makedirs(result_dir, exist_ok=True)
    output_name = os.path.splitext(os.path.basename(weights_path))[0] + ".txt"
    return os.path.join(result_dir, output_name)



def main(args):
    device = args.device
    input_keys = get_registration_input_keys(args.use_ct_input)

    model = SwinUNETR(
        in_channels=len(input_keys),
        out_channels=3,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        downsample="mergingv2",
        use_v2=True,
    )
    print(f'>>> Model input = {list(input_keys)}')

    train_transform = ReadH5d()

    train_list, test_list = split_multiple_train_test(
        ['/data2/xiangcen/data/pet_gen/processed/batch1_h5_v2','/data2/xiangcen/data/pet_gen/processed/batch2_h5_v2'],
        [40, 40]
    )


    test_loader = create_data_loader(
        test_list, train_transform, shuffle=False
    )

    identity_grid = make_identity_grid_m11(
        (128, 128, 384), device=device
    )

    if os.path.isdir(args.weights_path):
        
        weights_paths_list = [os.path.join(args.weights_path, _dir) for _dir in os.listdir(args.weights_path)]
        

        for weights_path in weights_paths_list:
            # get the file name
            filename = make_output_path(weights_path)
            print(f'txt saved at {filename}')

            if os.path.exists(filename):
                continue

            model.load_state_dict(torch.load(weights_path, map_location=device))
            inference_batch(
                model,
                test_loader,
                identity_grid,
                filename=filename,
                device=args.device,
                input_keys=input_keys,
            )
    else:
        filename = make_output_path(args.weights_path)
        print(f'txt saved at {filename}')

        model.load_state_dict(torch.load(args.weights_path, map_location=device))
        inference_batch(
            model,
            test_loader,
            identity_grid,
            filename=filename,
            device=args.device,
            input_keys=input_keys,
        )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="print results"
    )

    parser.add_argument(
        "--weights_path",
        type=str,
        required=True,
        help="Path to the trained SwinUNETR model weights (.pth)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run on (e.g., cuda, cuda:0, cpu)"
    )

    parser.add_argument(
        "--use_ct_input",
        action="store_true",
        help="Use [fdg_pt, fdg_ct, psma_pt, psma_ct] as model input"
    )



    args = parser.parse_args()

    main(args)
