call %INSTALLER_DIR%\activate_miniconda3_wrapper.sh
if errorlevel 1 exit /b
FOR /F "tokens=*" %%i in ('type tmp.txt') do SET %%i
