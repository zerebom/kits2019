import subprocess
 

data=r'C:\Users\higuchi\Desktop\kits19\data\case_00'
ct=r'\imaging.nii.gz'
seg=r'\segmentation.nii.gz'
for i in range (1,211):
    cmd = f"python ./scripts/extractSlices.py {data+str(i).zfill(3)+ct} {data+str(i).zfill(3)+seg} -o ctlistfile.txt"
    subprocess.call(cmd.split())

    # cmd = f"python ./scripts/extractSlices.py {data+str(i).zfill(3)+seg} -o seglistfile.txt"
    # subprocess.call(cmd.split())