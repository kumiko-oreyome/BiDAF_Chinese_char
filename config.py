NAME = "Chinese machine reading comprehension , character based (no word segmentation)"
DESCRIPTION = ""

TRAIN_FILE = '20.json'
DEV_FILE = '5.json'

PRE_TRAIN_FILE = TRAIN_FILE+'l'
PRE_DEV_FILE = DEV_FILE+'l'

CHAR_CHANNEL_NUM = 100
WORD_DIM = 100