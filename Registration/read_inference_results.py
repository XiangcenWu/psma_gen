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


def get_metrics_for_masks(mask_names_input, masks_names, dice_before_lists, dice_after_lists, tre_before_lists, tre_after_lists):
    """
    Returns the registration metrics for the given list of mask names.

    Parameters:
        mask_names_input (list of str): List of mask names to query.
        masks_names (list of str): Full list of all mask names.
        dice_before_lists (list of float): Dice scores before registration.
        dice_after_lists (list of float): Dice scores after registration.
        tre_before_lists (list of float): TRE values before registration.
        tre_after_lists (list of float): TRE values after registration.

    Returns:
        dict: Dictionary where keys are mask names and values are dicts of metrics.
    """
    results = {}
    for name in mask_names_input:
        if name in masks_names:
            idx = masks_names.index(name)
            results[name] = {
                "dice_before": dice_before_lists[idx],
                "dice_after": dice_after_lists[idx],
                "tre_before": tre_before_lists[idx],
                "tre_after": tre_after_lists[idx]
            }
        else:
            results[name] = {
                "dice_before": None,
                "dice_after": None,
                "tre_before": None,
                "tre_after": None
            }
    return results

if __name__ == "__main__":
    from pprint import pprint
    _dir = r"C:\Users\Sam\Downloads\pet_reg_results\ctsmoothness_l8000_k10_mar3000_gam1.2.txt"

    masks_names, dice_before_lists, dice_after_lists, tre_before_lists, tre_after_lists = load_registration_results(_dir)
    print(masks_names)
    # Example usage:
    organ = 'iliopsoas_right'
    query_names = [organ]
    data = get_metrics_for_masks(query_names, masks_names, dice_before_lists, dice_after_lists, tre_before_lists, tre_after_lists)


    import matplotlib.pyplot as plt
    import seaborn as sns

    # Combine data for boxplot
    metrics = ['Dice', 'TRE']
    before = [data[organ]['dice_before'], data[organ]['tre_before']]
    after = [data[organ]['dice_after'], data[organ]['tre_after']]

    # Set up figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sns.boxplot(data=before + after, ax=axes[0])
    axes[0].set_xticklabels(['Dice Before', 'TRE Before', 'Dice After', 'TRE After'])
    axes[0].set_title('Box Plot of Metrics Before and After')
    axes[0].set_ylabel('Values')

    # Optional: Separate plots for clarity
    axes[1].boxplot([data[organ]['dice_before'], data[organ]['dice_after']], labels=['Dice Before', 'Dice After'])
    axes[1].set_title('Dice Before vs After')
    axes[1].set_ylabel('Dice Score')

    plt.tight_layout()
    plt.show()
