################################## HPARAMS ##############################################
EPOCHS=10
NET_DEPTH=2
#########################################################################################

python3 main.py \
  --epochs ${EPOCHS} \
  --net_depth ${NET_DEPTH}