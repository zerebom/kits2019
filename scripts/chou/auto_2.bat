@echo off  
SETLOCAL ENABLEDELAYEDEXPANSION
::data path

set data=C:\Users\higuchi\Desktop\kits19\data\case_00
::file name
set ct=\imaging.nii.gz
set seg=\segmentation.nii.gz
::patch size
set psize=36x36x28


:: NO 15 and 25
REM set numArr=006,007,008,009,010,011,012,013,014,016,017,018,019,020,021,022,023,024,026,027

for /l %%i in (1,1,210) do (
    set num =000%%i
    REM set num=!num:~-3!
    

    python extractSlices.py %data%!num!%ct% 

)