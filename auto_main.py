import subprocess


# def call(cmd):
#     for i in range(1,4):
#         cmd2=''
#         cmd2=f"python main.py -d {i}"+cmd
#         subprocess.call(cmd2.split())

def call(cmd):
	cmd2='python3 main.py'+cmd
	subprocess.call(cmd2.split())

def pred(cmd):
	cmd2='python3 predict.py'+cmd
	subprocess.call(cmd2.split())


call(cmd = " -s -op -sp 1 -c 3 -b 16 -l 7 -f 48 -sf unet7_f48_c3_2")
# call(cmd = " -s -op -sp 1 -c 5 -b 16  -sf unet7_c5")
# call(cmd = " -e 1 -s -op -sp 1 -c 3 -b 32  -sf unet7_c3")
# call(cmd = " -s -op -sp 1 -c 3 -b 32 -f 32  -sf unet8_c3f32")
# call(cmd = " -s -op -sp 1 -c 3 -b 32   -sf unet8_c3f48")
# call(cmd = " -s -op -sp 1 -c 3 -b 32 -f 64　 -sf unet8_c3f64")

# call(cmd = " -s -op -sp 1 -c 3 -l 4 -b 48 -f 64　 -sf unet4_c3f64")
# call(cmd = " -s -op -sp 1 -c 1 -l 4 -b 48 -f 64　 -sf unet4_c1f64")
# call(cmd = " -s -op -sp 1 -c 1 -l 5 -b 48 -f 64　 -sf unet5_c1f64")
# call(cmd = " -s -op -sp 1 -c 3 -l 5 -b 48 -f 64　 -sf unet5_c3f64")



# call(cmd = " -s -op -sp 1 -c 1 -b 64  -sf unet7_c1")
# call(cmd = " -s -op -sp 1 -c 1 -b 64   -sf unet8_c1")


# pred(cmd=" -p 0716_unet8_c1.h5 -c 1")
# pred(cmd=" -p 0716_unet7_c5.h5 -c 5")
# pred(cmd=" -p 0716_unet7_c3.h5 -c 3")
# pred(cmd=" -p 0716_unet7_c1.h5 -c 1")






# call(cmd = " -s -op -sp 1 -c 1 -b 64   -f 48 -sf unet8_f48_c1")

# call(cmd = " -s -op -sp 1 -c 1 -b 64 -as  -f 48 -sf unet7_f48_as_c1")
# call(cmd = " -s -op -sp 1 -c 1 -b 64 -as -sf unet7_as_c1")
# call(cmd = " -s -op -sp 1 -c 1 -b 64 -as  -sf unet8_as_c1")
# call(cmd = " -s -op -sp 1 -c 1 -b 64 -as  -f 48 -sf unet8_f48_as_c1")



# call(cmd = " -s -op -sp 1 -c 3 -b 32   -f 48 -sf unet8_f48_c3")

# call(cmd = " -s -op -sp 1 -c 3 -b 32 -as  -f 48 -sf unet7_f48_as_c3")
# call(cmd = " -s -op -sp 1 -c 3 -b 32 -as -sf unet7_as_c3")
# call(cmd = " -s -op -sp 1 -c 3 -b 32 -as  -sf unet8_as_c3")
# call(cmd = " -s -op -sp 1 -c 3 -b 32 -as  -f 48 -sf unet8_f48_as_c3")

