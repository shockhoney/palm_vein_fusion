import os
import shutil
'''
提取指定文件夹中包含"WHT"的图像文件到目标文件夹。
'''

def extract_images(source_folder, target_folder):

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp', '.ico'}
    if not os.path.exists(source_folder):
        print(f"错误: 源文件夹不存在: {source_folder}")
        return
    os.makedirs(target_folder, exist_ok=True)
    print(f"目标文件夹: {target_folder}")
    
    total_found = 0
    total_extracted = 0
    failed = 0 
  
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            if file_ext in image_extensions and 'WHT' in file: 
                total_found += 1
                
                try:
                    target_path = os.path.join(target_folder, file)
                    counter = 1
                    original_target = target_path
                    while os.path.exists(target_path):
                        name, ext = os.path.splitext(original_target)
                        target_path = f"{name}_{counter}{ext}"
                        counter += 1
                    shutil.copy2(file_path, target_path)
                    total_extracted += 1
                    
                except Exception as e:
                    print(f"处理文件失败 {file}: {e}")
                    failed += 1
    
def main():  
    source_folder = r"E:\文档\学校文档\MMIF-CDDFuse-main\data\images"
    target_folder = r"E:\文档\学校文档\MMIF-CDDFuse-main\data\vis"
    extract_images(source_folder, target_folder)

if __name__ == "__main__":
    main()
