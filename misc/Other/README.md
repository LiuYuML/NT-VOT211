# Annotation Tool

## Start
To initiate the annotation process, run the following command:

```bash
python label_tool_ver_1.7f.py
```
## Select Folder
Select a folder that contains the video files.

## Basic Functions

### Navigation
- Press `a` to go to the previous frame.
- Press `d` to go to the next frame.

### Drawing Rectangle
Click and drag the left mouse button to draw a rectangular area.

### Clearing Annotations
Click `clear` to clear the annotation for the current frame.

### Jumping to a Specific Frame
Input a number and click `jump` to navigate to the desired frame.

### Marking Attributes
Click `out of view` or select `Occlusion` before drawing the rectangle to mark the attribute for the current frame.

## Finishing the Annotation for Current Video
Press `d` on the last frame or Click `skip` to complete the annotation for the current video.


# Meta information
We have provided calculated attributions for the following benchmarks:
- `NT-VOT211`
- `LaSOT`
- `Nfs`
- `AVisT`
- `OTB100`
- `DarkTrack2021`
- `NAT2021`
- `UAVDark135`

You can download the attributions from [here](https://github.com/LiuYuML/NV-VOT211/blob/main/misc/Other/meta.zip)

The scripts for attribution calculation can be found [here](https://github.com/LiuYuML/NV-VOT211/tree/main/misc/Other/scripts)
