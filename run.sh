# location for results
ROOT=$(git rev-parse --show-toplevel)
RESULTS_ROOT="${ROOT}/results"
mkdir -p ${RESULTS_ROOT}

## local variables
DATA_RAW="${ROOT}/data_raw"
DATA_PREP="${ROOT}/data_prepared"
TEST_GOLD="${DATA_RAW}/test.new"
TEST_PRED="${RESULTS_ROOT}/model_translations.txt"
CHECKPOINT_PATH="${RESULTS_ROOT}/checkpoint_best.pt"


#################
# Preprocessing #
#################
# python preprocess.py --data-raw  "${DATA_RAW}"  \
#                      --data-prep "${DATA_PREP}" \
#                      --log-file  "${RESULTS_ROOT}/log.out"


############
# Training #
############
python train.py --data "${DATA_PREP}"                \
                --save-dir "${RESULTS_ROOT}"         \
                --log-file "${RESULTS_ROOT}/log.out" \
                # --train-on-tiny


##############
# Prediction #
##############
python translate.py --data "${DATA_PREP}"                   \
                    --checkpoint-path  "${CHECKPOINT_PATH}" \
                    --translate-output "${TEST_PRED}"

##############
# BLEU Score #
##############
python bleu.py --test_gold ${TEST_GOLD} \
               --test_pred ${TEST_PRED} \
               --bleu-output "${RESULTS_ROOT}/bleu.txt"
