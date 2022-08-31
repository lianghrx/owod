# Task 1
python tools/train_net.py --num-gpus 8 --resume --config-file ./configs/OWOD_TOOLKIT/t1/t1_train.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OUTPUT_DIR "./output/t1"

# No need to finetune in Task 1, as there is no incremental component.

python tools/train_net.py --num-gpus 8 --config-file ./configs/OWOD_TOOLKIT/t1/t1_val.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OWOD.TEMPERATURE 1.5 OUTPUT_DIR "./output/t1_final" MODEL.WEIGHTS "./output/t1/model_final.pth"

python tools/train_net.py --num-gpus 8 --eval-only --config-file ./configs/OWOD_TOOLKIT/t1/t1_test.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/t1_final" MODEL.WEIGHTS "./output/t1/model_final.pth"


## Task 2
#mkdir -p ./output/t2 && cp -r ./output/t1/* ./output/t2/
#
#python tools/train_net.py --num-gpus 8 --resume --config-file ./configs/OWOD_TOOLKIT/t2/t2_train.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OUTPUT_DIR "./output/t2" MODEL.WEIGHTS "./output/t2/model_final.pth"
#
#mkdir -p ./output/t2_ft && cp -r ./output/t2/* ./output/t2_ft/
#
#python tools/train_net.py --num-gpus 8 --resume --config-file ./configs/OWOD_TOOLKIT/t2/t2_ft.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OUTPUT_DIR "./output/t2_ft" MODEL.WEIGHTS "./output/t2_ft/model_final.pth"
#
#python tools/train_net.py --num-gpus 8 --config-file ./configs/OWOD_TOOLKIT/t2/t2_val.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OWOD.TEMPERATURE 1.5 OUTPUT_DIR "./output/t2_final" MODEL.WEIGHTS "./output/t2_ft/model_final.pth"
#
#python tools/train_net.py --num-gpus 8 --eval-only --config-file ./configs/OWOD_TOOLKIT/t2/t2_test.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/t2_final" MODEL.WEIGHTS "./output/t2_ft/model_final.pth"
#
#
## Task 3
#mkdir -p ./output/t3 && cp -r ./output/t2_ft/* ./output/t3/
#
#python tools/train_net.py --num-gpus 8 --resume --config-file ./configs/OWOD_TOOLKIT/t3/t3_train.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OUTPUT_DIR "./output/t3" MODEL.WEIGHTS "./output/t3/model_final.pth"
#
#mkdir -p ./output/t3_ft && cp -r ./output/t3/* ./output/t3_ft/
#
#python tools/train_net.py --num-gpus 8 --resume --config-file ./configs/OWOD_TOOLKIT/t3/t3_ft.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OUTPUT_DIR "./output/t3_ft" MODEL.WEIGHTS "./output/t3_ft/model_final.pth"
#
#python tools/train_net.py --num-gpus 8 --config-file ./configs/OWOD_TOOLKIT/t3/t3_val.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OWOD.TEMPERATURE 1.5 OUTPUT_DIR "./output/t3_final" MODEL.WEIGHTS "./output/t3_ft/model_final.pth"
#
#python tools/train_net.py --num-gpus 8 --eval-only --config-file ./configs/OWOD_TOOLKIT/t3/t3_test.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/t3_final" MODEL.WEIGHTS "./output/t3_ft/model_final.pth"
