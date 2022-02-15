
if [ ! -d "output" ]; then
    mkdir "output"
fi

for model in 'rscnn' 'pct' 'pointnet' 'pointnet2'  'simpleview' 'dgcnn'; do
for cor in 'uniform' 'gaussian' 'background' 'impulse' 'upsampling' 'distortion_rbf' 'distortion_rbf_inv' 'density' 'density_inc' 'shear' 'rotation' 'cutout' 'distortion'  'occlusion' 'lidar'; do

for sev in 1 2 3 4 5; do

CUDA_VISIBLE_DEVICES=0 python main.py --entry test --model-path runs/cutmix_r_${model}_run_1/model_best_test.pth --exp-config configs/tent_cutmix/${model}.yaml --severity ${sev} --corruption ${cor} --output ./output/${model}_megamerger_${cor}_${sev}.txt 

done
done
done
