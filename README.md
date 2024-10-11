# AI Programming with Python Project

Hello I am Rashmi Maurya , a AWS AI & ML Scholar'24. This repository contains my final project I completed as a part of Udacity's AI Programming with Python Nanodegree program. In this, I first develop code for an image classifier built with PyTorch, then convert it into a command line application. This project really helped me learn alot in the AI field using python programming.




#How to run scripts in command line

#For running train.py script following command can be used in terminal:
```!python train.py flowers --save_dir checkpoints --arch vgg16 --learning_rate 0.001 --hidden_units 512 --epochs 5 --gpu  ```

#For running predict.py script following command can be used in terminal:
```!python predict.py flowers/test/1/image_06743.jpg checkpoints/checkpoint.pth --top_k 3 --category_names cat_to_name.json --gpu  ```


* You can change the parameters used accordingly.
* This format of command is for running scripts in Google Colab. You may remove "!" used if you are running it directly in terminal.


