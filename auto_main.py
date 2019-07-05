import subprocess


def call(cmd):
    for i in range(1,4):
        cmd2=''
        cmd2=f"python main.py -d {i}"+cmd
        subprocess.call(cmd2.split())

# call(cmd = " -s -ap   -f 48 -sf unet7_f48")
call(cmd = " -s -ap  -sf unet7")
call(cmd = " -s -ap  -ei -sf unet8")
call(cmd = " -s -ap  -ei -f 48 -sf unet8_f48")

call(cmd = " -s -ap -as  -f 48 -sf unet7_f48_as")
call(cmd = " -s -ap -as -sf unet7_as")
call(cmd = " -s -ap -as -ei -sf unet8_as")
call(cmd = " -s -ap -as -ei -f 48 -sf unet8_f48_as")

