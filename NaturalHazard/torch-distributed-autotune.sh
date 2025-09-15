clear
echo "Training Vgg16 model in distributed fashion..."
torchrun --nproc_per_node 4 torch-train-3rd-distributed-autotune.py
