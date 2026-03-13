import SimpleITK as sitk


def tensor_to_itk(tensor, spacing=None):
    """
    tensor: (1, 1, X, Y, Z) 
    """

    tensor = tensor.cpu()[0, 0]

    array = tensor.numpy().transpose(2, 1, 0)
    itk_img = sitk.GetImageFromArray(array)

    if spacing is not None:
        itk_img.SetSpacing(spacing)


    return itk_img
