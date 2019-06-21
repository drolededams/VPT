import os

subject = 1
while subject <= 108:
    task = 1
    print("subject", subject)
    while task <= 4:
        print("task", task)
        os.system("python3 train_TPV.py -s {} -t {} ".format(subject, task))
        os.system("python3 train_TPV.py -s {} -t {} predict ".format(subject, task))
        os.system("python3 train_TPV.py -s {} -t {} rtpredict ".format(subject, task))
        os.system("python3 train_TPV.py -s {} -t {} rtpredict -pref ".format(subject, task))
        os.system("python3 train_TPV.py -s {} -t {} -f psd ".format(subject, task))
        os.system("python3 train_TPV.py -s {} -t {} predict ".format(subject, task))
        os.system("python3 train_TPV.py -s {} -t {} rtpredict ".format(subject, task))
        os.system("python3 train_TPV.py -s {} -t {} rtpredict -pref ".format(subject, task))
        print("task end", task)
        task += 1
    subject += 1
