
###
 # @Description: 
 # @Autor: Jiachen Sun
 # @Date: 2022-01-27 17:01:16
 # @LastEditors: Jiachen Sun
 # @LastEditTime: 2022-01-27 17:01:38
### 
for model in 'rscnn' 'pct' 'pointnet' 'pointnet2'  'simpleview' 'dgcnn'; do
for cor in 'uniform' 'gaussian' 'background' 'impulse' 'upsampling' 'distortion_rbf' 'distortion_rbf_inv' 'density' 'density_inc' 'shear' 'rotation' 'cutout' 'distortion'  'occlusion' 'lidar'; do

for sev in 1 2 3 4 5; do

for aug in 'rsmix' 'cutmix_r' 'cutmix_k' 'mixup' 'pgd'; do

CUDA_VISIBLE_DEVICES=0 python main.py --entry test --model-path runs/${aug}_${model}_run_1/model_best_test.pth --exp-config configs/corruption/${model}.yaml --severity ${sev} --corruption ${cor} --output ./output/${model}_${aug}_${cor}_${sev}.txt

done

for adapt in 'tent' 'bn'; do

CUDA_VISIBLE_DEVICES=0 python main.py --entry test --model-path cor_exp/dgcnn_${model}_run_1/model_best_test.pth --exp-config configs/${adapt}/${model}.yaml --severity ${sev} --corruption ${cor} --output ./output/${model}_${adapt}_${cor}_${sev}.txt

done

CUDA_VISIBLE_DEVICES=0 python main.py --entry test --model-path cor_exp/dgcnn_${model}_run_1/model_best_test.pth --exp-config configs/corruption/${model}.yaml --severity ${sev} --corruption ${cor} --output ./output/${model}_none_${cor}_${sev}.txt

done
done
done
