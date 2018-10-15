# BiDAF_Chinese_char
Machine reading comprehension with BiDAF model 
Character based without word segmentation, span output
train on small data
remove EMA


Train

python main.py --batch-size=16  train --save-freq=5

Test
python  core.py --batch-size=1 test ./datas/5.json <run_directory>

Demo
python  main.py demo ./demo.txt <run_directory>

