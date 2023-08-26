# RANDOM
# CUDA_VISIBLE_DEVICES=2 python main.py --exp-config configs/pgd/dgcnn.yaml --pruning_rate 0.9
# CUDA_VISIBLE_DEVICES=2 python main.py --exp-config configs/pgd/dgcnn.yaml --pruning_rate 0.8
# CUDA_VISIBLE_DEVICES=2 python main.py --exp-config configs/pgd/dgcnn.yaml --pruning_rate 0.7
# CUDA_VISIBLE_DEVICES=2 python main.py --exp-config configs/pgd/dgcnn.yaml --pruning_rate 0.5
# CUDA_VISIBLE_DEVICES=2 python main.py --exp-config configs/pgd/dgcnn.yaml --pruning_rate 0.3

#FORGETTING
# python main.py --coreset_method forgetting --exp-config configs/pgd/dgcnn.yaml --pruning_rate 0.7 > forgetting.txt
# python main.py --coreset_method forgetting --exp-config configs/pgd/dgcnn.yaml --pruning_rate 0.5 >> forgetting.txt
# python main.py --coreset_method forgetting --exp-config configs/pgd/dgcnn.yaml --pruning_rate 0.3 >> forgetting.txt
# python main.py --coreset_method forgetting --exp-config configs/pgd/dgcnn.yaml --pruning_rate 0.1 >> forgetting.txt


#EL2N
# python main.py --coreset_method el2n --exp-config configs/pgd/dgcnn.yaml --pruning_rate 0.9 > el2n.txt
# python main.py --coreset_method el2n --exp-config configs/pgd/dgcnn.yaml --pruning_rate 0.8 > el2n.txt
# python main.py --coreset_method el2n --exp-config configs/pgd/dgcnn.yaml --pruning_rate 0.7 >> el2n.txt
# python main.py --coreset_method el2n --exp-config configs/pgd/dgcnn.yaml --pruning_rate 0.5 >> el2n.txt
# python main.py --coreset_method el2n --exp-config configs/pgd/dgcnn.yaml --pruning_rate 0.3 >> el2n.txt
# python main.py --coreset_method el2n --exp-config configs/pgd/dgcnn.yaml --pruning_rate 0.1 >> el2n.txt


#AUM
python main.py --coreset_method accumulated_margin --exp-config configs/pgd/dgcnn.yaml --pruning_rate 0.9 > aum.txt
python main.py --coreset_method accumulated_margin --exp-config configs/pgd/dgcnn.yaml --pruning_rate 0.8 >> aum.txt
python main.py --coreset_method accumulated_margin --exp-config configs/pgd/dgcnn.yaml --pruning_rate 0.7 >> aum.txt
python main.py --coreset_method accumulated_margin --exp-config configs/pgd/dgcnn.yaml --pruning_rate 0.5 >> aum.txt
python main.py --coreset_method accumulated_margin --exp-config configs/pgd/dgcnn.yaml --pruning_rate 0.3 >> aum.txt
# python main.py --coreset_method el2n --exp-config configs/pgd/dgcnn.yaml --pruning_rate 0.1 >> el2n.txt