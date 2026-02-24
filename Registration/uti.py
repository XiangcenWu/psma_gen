
import numpy as np
from scipy import stats
import ast
def load_registration_results(filename):
    """
    Load registration results from text file.

    Returns:
        masks_names,
        dice_before_lists,
        dice_after_lists,
        tre_before_lists,
        tre_after_lists
    """

    def parse_list_line(line):
        parts = line.strip().split(";")
        parsed = []
        for p in parts:
            p = p.strip()

            # try to interpret as python literal (list, float, etc.)
            try:
                parsed.append(ast.literal_eval(p))
            except Exception:
                parsed.append(p)

        return parsed

    with open(filename, "r") as f:
        lines = f.readlines()

    masks_names = parse_list_line(lines[0])
    dice_before_lists = parse_list_line(lines[1])
    dice_after_lists = parse_list_line(lines[2])
    tre_before_lists = parse_list_line(lines[3])
    tre_after_lists = parse_list_line(lines[4])

    return (
        masks_names,
        dice_before_lists,
        dice_after_lists,
        tre_before_lists,
        tre_after_lists,
    )


def print_p(listA, listB, organ_name):

    model_A = np.asarray(ast.literal_eval(listA), dtype=float)  # length = 80
    model_B = np.asarray(ast.literal_eval(listB), dtype=float)  # length = 80

    print(model_A)

    t_stat, p_value = stats.ttest_rel(model_A, model_B)
    print(organ_name)
    print("Paired t-test")
    print("t-statistic:", t_stat)
    print("p-value:", p_value)