@echo off
REM Run after ComfyUI is fully quit. Prevents duplicate MiniMax H3 node IDs.
cd /d "%~dp0.."
if exist "ComfyUI-MiniMaxH3-TimelineDirector.disabled" (
  echo Already disabled.
  exit /b 0
)
if not exist "ComfyUI-MiniMaxH3-TimelineDirector" (
  echo Original Timeline Director folder not found. Nothing to do.
  exit /b 0
)
ren "ComfyUI-MiniMaxH3-TimelineDirector" "ComfyUI-MiniMaxH3-TimelineDirector.disabled"
if errorlevel 1 (
  echo Rename failed. Close ComfyUI completely and run this again.
  exit /b 1
)
echo Disabled original Timeline Director.
