import SimpleITK as sitk
import os
from collections import defaultdict

def get_slice_location(filepath):
    """获取DICOM切片的空间位置信息"""
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(filepath)
        reader.ReadImageInformation()
        
        # 优先使用Instance Number
        if reader.HasMetaDataKey("0020|0013"):
            instance_num = int(reader.GetMetaData("0020|0013"))
            return instance_num
        
        # 其次使用Image Position Patient的Z坐标
        elif reader.HasMetaDataKey("0020|0032"):
            position = reader.GetMetaData("0020|0032")
            z_pos = float(position.split('\\')[2])
            return z_pos
            
        # 最后使用Slice Location
        elif reader.HasMetaDataKey("0020|1041"):
            slice_loc = float(reader.GetMetaData("0020|1041"))
            return slice_loc
        
        return 0
    except:
        return 0


def separate_dicom_modalities(input_folder, output_folder, pet_tracer):
    """
    读取混合的DICOM文件并按模态分离(修复版)
    
    参数:
        input_folder: 包含混合DICOM文件的文件夹路径
        output_folder: 输出文件夹路径
    """
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 存储不同模态的文件
    modality_files = defaultdict(list)
    
    # 遍历所有DICOM文件
    print("正在扫描DICOM文件...")
    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)
        
        # 跳过非文件
        if not os.path.isfile(filepath):
            continue
        
        try:
            # 读取DICOM文件的元数据
            reader = sitk.ImageFileReader()
            reader.SetFileName(filepath)
            reader.LoadPrivateTagsOn()
            reader.ReadImageInformation()
            
            # 获取模态信息
            if reader.HasMetaDataKey("0008|0060"):
                modality = reader.GetMetaData("0008|0060")
            else:
                modality = "Unknown"
            
            # 获取SeriesInstanceUID用于区分不同序列
            if reader.HasMetaDataKey("0020|000e"):
                series_uid = reader.GetMetaData("0020|000e")
            else:
                series_uid = "unknown_series"
            
            # 使用模态和序列UID组合作为键
            key = f"{modality}_{series_uid}"
            modality_files[key].append(filepath)
            
        except Exception as e:
            print(f"无法读取文件 {filename}: {e}")
            continue
    
    # 按模态分组并读取图像
    print(f"\n找到 {len(modality_files)} 个不同的序列:")
    
    for idx, (key, files) in enumerate(modality_files.items(), 1):
        modality = key.split('_')[0]
        print(f"\n序列 {idx}: {modality} - {len(files)} 个文件")
        
        try:
            # ⭐ 关键修复: 按切片位置排序,而不是文件名!
            print("  正在按切片位置排序...")
            files_with_location = [(f, get_slice_location(f)) for f in files]
            files_sorted = [f for f, _ in sorted(files_with_location, key=lambda x: x[1])]
            
            # 使用SimpleITK的自动排序功能(推荐!)
            reader = sitk.ImageSeriesReader()
            
            # 方法1: 让SimpleITK自动排序(最可靠)
            series_file_names = reader.GetGDCMSeriesFileNames(
                input_folder, 
                key.split('_', 1)[1]  # 使用series UID
            )
            
            if len(series_file_names) > 0:
                reader.SetFileNames(series_file_names)
            else:
                # 如果自动排序失败,使用我们手动排序的结果
                reader.SetFileNames(files_sorted)
            
            image = reader.Execute()
            

            
            # 保存为.nii.gz格式
            output_path = os.path.join(output_folder, pet_tracer + f"{modality}_series{idx}.nii.gz")
            sitk.WriteImage(image, output_path)
            print(f"  ✓ 已保存: {output_path}")
            print(f"  图像尺寸: {image.GetSize()}")
            print(f"  图像间距: {image.GetSpacing()}")
            print(f"  切片数量: {image.GetDepth()}")
            
        except Exception as e:
            print(f"  ✗ 处理序列 {key} 时出错: {e}")
            print(f"    尝试使用手动排序...")
            
            # 备用方案: 手动排序
            try:
                files_with_location = [(f, get_slice_location(f)) for f in files]
                files_sorted = [f for f, _ in sorted(files_with_location, key=lambda x: x[1])]
                
                reader = sitk.ImageSeriesReader()
                reader.SetFileNames(files_sorted)
                image = reader.Execute()
                

                output_path = os.path.join(output_folder, pet_tracer + f"{modality}_series{idx}.nii.gz")
                sitk.WriteImage(image, output_path)
                print(f"  ✓ 备用方案成功!")
                
            except Exception as e2:
                print(f"  ✗ 备用方案也失败: {e2}")
                continue
    
    print(f"\n完成! 结果保存在: {output_folder}")