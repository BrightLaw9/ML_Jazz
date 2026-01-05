SEQ_LEN = 128
CHECKPOINT_FREQ = 10
MODEL_CHECKPOINT_DIR = "checkpoints/model/"
OPTIMIZER_CHECKPOINT_DIR = "checkpoints/optimizer/"

TRAIN_DIR = "train_midi_test/"

INFERENCE_MODEL = "model_ckpt_1001.pth" #"trained.pth"
#INITIAL_TOKENS_PATH = "train_midi_pt_dur/JAZZ_456_DOWNLOADS/CharlieParker_BluesForAlice_FINAL.pt"
# INITIAL_TOKENS_PATH = "train_midi_pt_dur/JAZZ_456_DOWNLOADS/ChetBaker_ThereWillNeverBeAnotherYou-1_FINAL.pt"
#INITIAL_TOKENS_PATH = "train_midi_pt_dur/JAZZ_456_DOWNLOADS/CharlieParker_BluesForAlice_FINAL.pt"
INITIAL_TOKENS_PATH = "train_midi_pt_dur/Jazz MIDI 3/AutumnLeaves.pt"
GENERATED_SAVE_DIR = "transformer_gen/"

D_MODEL = 512
NUM_LAYERS = 8
NHEAD = 8
DIM_FF = 4096

CKPT_MODEL = "" #"model_ckpt_977.pth"
CKPT_OPTIM = "" #"optim_ckpt_977.pth"
EPOCH_NUM = 0 # 250 # 0
NUM_EPOCHES = 1500
LEARN_RATE = 1e-4

NOTE_ON_OFFSET = 0
DUR_OFFSET = 128

MAX_DUR_SHIFT_MS = 2000
DUR_RES_MS = 2
MAX_DUR_TOKENS = MAX_DUR_SHIFT_MS // DUR_RES_MS

TIME_SHIFT_OFFSET = DUR_OFFSET + MAX_DUR_TOKENS

MAX_TIME_SHIFT_MS = 1000
TIME_RES_MS = 5
MAX_TIME_TOKENS = (MAX_TIME_SHIFT_MS // TIME_RES_MS) + 1 # 100

VELOCITY_OFFSET = TIME_SHIFT_OFFSET + MAX_TIME_TOKENS #456
MAX_VEL_TOK = 100

VOCAB_SIZE = VELOCITY_OFFSET + MAX_VEL_TOK + 1 # exclusive

def bin_to_velocity(vel):
    return min(128 * vel // 32, MAX_VEL_TOK * 128 // 32)
