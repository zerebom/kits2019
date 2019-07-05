import subprocess



for i in range(1,4):
    # cmd = f"python main.py -e 1 -d {i}"
    cmd = f"python main.py -d {i}  -f 48 -sf unet7_f48"
    subprocess.call(cmd.split())
    
    cmd = f"python main.py -d {i} -sf -sf unet7"
    subprocess.call(cmd.split())
    
    cmd = f"python main.py -d {i} -ei -sf unet8"
    subprocess.call(cmd.split())

    cmd = f"python main.py -d {i} -ei -f 48 -sf unet8_f48"
    subprocess.call(cmd.split())








    # cmd = f"python ./scripts/extractSlices.py {data+str(i).zfill(3)+seg} -o seglistfile.txt"
    # subprocess.call(cmd.split())
