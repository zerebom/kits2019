import subprocess



for i in range(1,4):
    # cmd = f"python main.py -e 1 -d {i}"

    cmd = f"python main.py -d {i}"
    subprocess.call(cmd.split())
    
    cmd = f"python main.py -d {i} -ei"
    subprocess.call(cmd.split())




    # cmd = f"python ./scripts/extractSlices.py {data+str(i).zfill(3)+seg} -o seglistfile.txt"
    # subprocess.call(cmd.split())
