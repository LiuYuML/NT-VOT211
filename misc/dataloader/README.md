# Tutorial for evaluation script

## Unzip the ZIP File

To unzip the file, execute the following bash command:

```bash
bash NT-VOT211_Unziper.sh
```

**Before running this command, ensure you have configured the following:**

1. **Line 4:** Set the `ZIP_FILE_PATH` variable to the path of your ZIP file.
   ```bash
   ZIP_FILE_PATH="/path/to/NT-VOT211.zip"
   ```

2. **Line 7:** Define the `TARGET_PATH` variable to specify the directory where the contents of the ZIP file should be extracted.
   ```bash
   TARGET_PATH="/path/to/target"
   ```

**Example Configuration:**

If your ZIP file is located at `/home/user/downloads/NT-VOT211.zip` and you want to extract it to `/home/user/projects/`, your script should look like this:

```bash
#!/bin/bash

# Path to the ZIP file
ZIP_FILE_PATH="/home/user/downloads/NT-VOT211.zip"

# Target directory for extraction
TARGET_PATH="/home/user/projects/"
```


## add the necessary files
If you are using the official [pytracking](https://github.com/visionml/pytracking/tree/master) framework, please add the provided `nt_vot_211dataset.py` file to the `lib/test/evaluation` directory. For those using an adapted version of the pytracking framework, such as [ARTrack](https://github.com/MIV-XJTU/ARTrack), the file should be placed in the `pytracking/evaluation` directory. This directory usually looks like this:
![image](https://github.com/user-attachments/assets/8bf7bbab-360c-4d6d-8707-291df8d403e1)

## revise the content of dataset.py file
In the same directory where you placed the `nt_vot_211dataset.py` file, locate the `dataset.py` file. On line 10, please add the following code:
```python
ntvot211 = DatasetInfo(module=pt % "nt_vot_211", class_name="NT_VOT211Dataset", kwargs=dict()),
```
After the proper revision, the content should appear as the image below:
![image](https://github.com/user-attachments/assets/0398cc28-d59b-4581-a1bc-9983fc158e74)

## revise the content of local.py file
In the same directory where you placed the `nt_vot_211dataset.py` file, locate the `local.py` file. please add the following code:
```python
settings.nt_vot211_path = '<path-to-your-NT-VOT211 dataset>'
```
## run the evaluation
When executing the original test command, replace the existing `--dataset xxx` argument with `--dataset ntvot211`.
