## script for training on servers
echo "Pretraining..."
sleep 3

nohup venv/bin/python pretrain.py

tail -f nohup.out
