import json
import os
import glob

input_dir = 'be/data/cameras/awl'
file_paths = glob.glob(os.path.join(input_dir, 'extrinsic_*.json'))

for file_path in file_paths:
    with open(file_path, 'r') as f:
        data = json.load(f)
    print("Found file:", file_path)
    if 'P' in data:
        P = data['P']
        H = data['Hw2i']
            
        new_data = {
            "camera projection matrix": P,
            "homography matrix": H,
            "reprojection_error": 0.0
        }
        
        cid = data.get('cid')
        if not cid:
            filename = os.path.basename(file_path)
            cid = filename.split('_')[1].split('.')[0].replace('c', '')
            
        out_filename = f"cam{cid}.json"
        out_path = os.path.join(input_dir, out_filename)
        
        with open(out_path, 'w') as f:
            json.dump(new_data, f, indent=4)
            
        print(f"✅ Đã convert: {file_path} -> {out_path}")

print("Hoàn tất!")
