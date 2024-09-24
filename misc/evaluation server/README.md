# Tutorial for submission
## Register an account
To participate in our competition, hosted on [EvalAI](https://eval.ai), you'll need to have an account. Please sign up to join the challenge.

## Find our challenge
Once you have an `EvalAI` account, you can access our challenge through this [link](https://eval.ai/web/challenges/challenge-page/2375/).

## Transform your raw results into .json file
We've provided a `converter.py` script to assist you in converting your raw results into a `.json` file. To use it, simply run the following command:

```bash
python converter.py
```
Before executing the script, ensure that you have correctly set the variables on lines 4, 5, and 6.

## (IMPORTANT) choose your phase and submit
Please note that our challenge is divided into two phases: the **public phase** and the **private phase**.
- In the **private phase**, you are allowed to submit a maximum of **100 submissions per day**.
- In the **public phase**, you are limited to a maximum of **1 submission per month**.
To maintain a clean and readable `Leaderboard` for everyone, we strongly encourage the challengers to submit their .json file in private phase. So that you can try different hyper-parameter setting. And only submit the most competitive results on our `Leaderboard`. For the public phase, challengers are also asked to fill the device(GPU) they use and the url of the corresponding publication, this is aimed to hlep the possible citation of your method. For the pbulic phase, if you want your evaluation on `Leaderboard`， please check "Baseline" in `My Submissions`
![image](https://github.com/user-attachments/assets/eb1241fd-2b38-4db3-a616-8c8714b21636)
