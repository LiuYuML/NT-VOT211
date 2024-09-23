# Tutorial for evaluation script
## add the necessary files
If you are using the official [pytracking](https://github.com/visionml/pytracking/tree/master) framework, please add the provided `nt_vot_211dataset.py` file to the `lib/test/evaluation` directory. For those using an adapted version of the pytracking framework, such as [ARTrack](https://github.com/MIV-XJTU/ARTrack), the file should be placed in the `pytracking/evaluation` directory. This directory usually looks like this:
![image](https://github.com/user-attachments/assets/8bf7bbab-360c-4d6d-8707-291df8d403e1)

## revise the content of .py file
In the same directory you put the `nt_vot_211dataset.py`, there is a `dataset.py`, please add this `ntvot211=DatasetInfo(module=pt % "nt_vot_211", class_name="NT_VOT211Dataset", kwargs=dict()),` to the line 10. A proper revision should make the content look like this:
![image](https://github.com/user-attachments/assets/0398cc28-d59b-4581-a1bc-9983fc158e74)
