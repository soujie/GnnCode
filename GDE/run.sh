echo '预处理ml100k'
python preprocess.py --dataset 'ml_100k' --smooth_ratio 0.2 --rough_ratio 0.02

echo 'train'
python main.py --dataset ml_100k --method 'both' --loss_type 'bpr'


echo '预处理pinterest'
python preprocess.py --dataset 'pinterest' --smooth_ratio 0.05 --rough_ratio 0 