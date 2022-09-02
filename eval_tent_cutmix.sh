
###
 # @Description: 
 # @Autor: Jiachen Sun
 # @Date: 2022-02-16 22:23:16
 # @LastEditors: Jiachen Sun
 # @LastEditTime: 2022-05-28 10:21:37
### 
if [ ! -d "output" ]; then
    mkdir "output"
fi

for model in 'gdanet'; do #'rscnn' 'pct' 'pointnet' 'pointnet2'  'simpleview' 'dgcnn'; do
for cor in 'upsampling'; do

for sev in 5; do

CUDA_VISIBLE_DEVICES=0 python main.py --entry test --model-path runs/cutmix_r_${model}_run_1/model_best_test.pth --exp-config configs/tent_cutmix/${model}.yaml --severity ${sev} --corruption ${cor} --output ./output/${model}_megamerger_${cor}_${sev}.txt 

done
done
done
