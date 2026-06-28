```
python train.py --model conv --size tiny  --on-the-fly --steps 3000  --batch-size 256  --log-csv conv_tiny.csv
python train.py --model conv --size small --on-the-fly --steps 15000 --batch-size 512  --log-csv conv_small.csv
python train.py --model conv --size paper --on-the-fly --steps 50000 --batch-size 1024 --optimizer muon-lion --ema --log-csv conv_paper.csv   # GPU

python train.py --model topo --size tiny  --on-the-fly --steps 3000  --batch-size 256  --log-csv topo_tiny.csv
python train.py --model topo --size small --on-the-fly --steps 15000 --batch-size 512  --log-csv topo_small.csv
python train.py --model topo --size paper --on-the-fly --steps 50000 --batch-size 1024 --optimizer muon-lion --ema --log-csv topo_paper.csv   # GPU


```
