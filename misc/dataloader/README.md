# Tutorial for evaluation script
## add the necessary files
If you are using the official [pytracking](https://github.com/visionml/pytracking/tree/master) framework, please add the provided `nt_vot_211dataset.py` file to the `lib/test/evaluation` directory. For those using an adapted version of the pytracking framework, such as [ARTrack](https://github.com/MIV-XJTU/ARTrack), the file should be placed in the `pytracking/evaluation` directory. This directory usually looks like this:
![image](https://github.com/user-attachments/assets/8bf7bbab-360c-4d6d-8707-291df8d403e1)

## revise the content of dataset.py file
In the same directory where you placed the `nt_vot_211dataset.py` file, locate the `dataset.py` file. On line 10, please add the following code:
```python
ntvot211 = DatasetInfo(module=pt % "nt_vot_211", class_name="NT_VOT211Dataset", kwargs=dict()),
```
After the proper revision, the content should appear as shown in the image below:
![image](https://github.com/user-attachments/assets/0398cc28-d59b-4581-a1bc-9983fc158e74)

## revise the content of local.py file
In the same directory where you placed the `nt_vot_211dataset.py` file, locate the `local.py` file. please add the following code:
```python
settings.davis_dir = '<path-to-your-NT-VOT211 dataset>'
```
