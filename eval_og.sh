if [ ! -d "output" ]; then
    mkdir "output"
fi

for model in 'dgcnn' 'rscnn' 'pct' 'pointnet' 'pointnet2'  'simpleview'; do
# for aug in 'pgd'; do

CUDA_VISIBLE_DEVICES=1 python main.py --entry test --model-path cor_exp/dgcnn_${model}_run_1/model_best_test.pth --exp-config configs/dgcnn_${model}_run_1.yaml --output ./output/${model}_clean.txt

done
