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
To maintain a clean and readable **Leaderboard** for everyone, we strongly encourage challengers to submit their **.json** files during the **private phase**. This allows you to experiment with different hyper-parameter settings. Please only submit your most competitive results on our **Leaderboard**.

For the **public phase**, challengers are also asked to provide the following information:
- The device (GPU) used for the submission.
- The URL of the corresponding publication.

This information facilitates the potential citation of your method.

If you wish your evaluation to appear on the **Leaderboard** during the public phase, please ensure that you check "Baseline" in **My Submissions**.
![image](https://github.com/user-attachments/assets/eb1241fd-2b38-4db3-a616-8c8714b21636)

# Draw your attribution-wise evaluation
As detailed in our paper, the evaluation server will also provide an attribution-wise evaluation. To access it, follow these steps:

1. Navigate to "My Submissions."
2. Select either "public" or "private" evaluation.
3. Click on the "Link" under the "Result file" column.

Here's what you'll see:
![Screenshot of Evaluation Server](https://github.com/user-attachments/assets/c9032777-3334-476a-a172-f4b7a836440f)

Copy the entire content, which should appear as follows:
![Content to Copy](https://github.com/user-attachments/assets/ac558b49-ae60-42bf-b50f-f9c9b3283b10)

Once copied, paste the data and save the result as `<Tracker Name>.json`. Ensure it is placed in the same directory as `attribute-wise drawing.py`.
