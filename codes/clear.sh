rm -f ../result/plots/*.png
rm -f ../result/output/result_*.txt
rm -f ../result/output/inferenced_*.json

# keep the *best* checkpoints undeleted
rm -f ../result/checkpoint/training*.ckpt
