import subprocess


# def call(cmd):
#     for i in range(1,4):
#         cmd2=''
#         cmd2=f"python main.py -d {i}"+cmd
#         subprocess.call(cmd2.split())

def call(cmd):
	cmd2='python3 main.py'+cmd
	subprocess.call(cmd2.split())


# call(cmd = " -s -op -c 3 -b 16   -f 48 -sf unet7_f48_c3")
call(cmd = " -s -op -sp 1 -c 5 -b 16  -sf unet7_c5")
call(cmd = " -s -op -sp 1 -c 3 -b 32  -sf unet7_c3")
call(cmd = " -s -op -sp 1 -c 3 -b 32  -ei -sf unet8_c3")
call(cmd = " -s -op -sp 1 -c 1 -b 64  -sf unet7_c1")
call(cmd = " -s -op -sp 1 -c 1 -b 64  -ei -sf unet8_c1")


# call(cmd = " -s -op -sp 1 -c 1 -b 64  -ei -f 48 -sf unet8_f48_c1")

# call(cmd = " -s -op -sp 1 -c 1 -b 64 -as  -f 48 -sf unet7_f48_as_c1")
# call(cmd = " -s -op -sp 1 -c 1 -b 64 -as -sf unet7_as_c1")
# call(cmd = " -s -op -sp 1 -c 1 -b 64 -as -ei -sf unet8_as_c1")
# call(cmd = " -s -op -sp 1 -c 1 -b 64 -as -ei -f 48 -sf unet8_f48_as_c1")



# call(cmd = " -s -op -sp 1 -c 3 -b 32  -ei -f 48 -sf unet8_f48_c3")

# call(cmd = " -s -op -sp 1 -c 3 -b 32 -as  -f 48 -sf unet7_f48_as_c3")
# call(cmd = " -s -op -sp 1 -c 3 -b 32 -as -sf unet7_as_c3")
# call(cmd = " -s -op -sp 1 -c 3 -b 32 -as -ei -sf unet8_as_c3")
# call(cmd = " -s -op -sp 1 -c 3 -b 32 -as -ei -f 48 -sf unet8_f48_as_c3")

