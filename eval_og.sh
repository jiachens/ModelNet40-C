
if [ ! -d "output" ]; then
    mkdir "output"
fi

for model in 'dgcnn' 'rscnn' 'pct' 'pointnet' 'pointnet2'  'simpleview'; do
for aug in 'pgd'; do

CUDA_VISIBLE_DEVICES=0 python main.py --entry test --model-path runs/${aug}_${model}_run_1/model_best_test.pth --exp-config configs/dgcnn_${model}_run_1.yaml --output ./output/${model}_${aug}_clean.txt

done
done
