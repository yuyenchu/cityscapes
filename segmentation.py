import argparse
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime
from clearml import Task, Dataset
from utils import *

DEBUG = False

# device capabilities
physical_devices = tf.config.list_physical_devices('GPU')
print("Tensorflow Version:",tf.__version__, ", GPU devices:",physical_devices)
if (len(physical_devices) > 0):
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        print('='*10,"Error setting TF GPU memory growth",'='*10)

def get_parser():
    parser = argparse.ArgumentParser(description='options for training')
    parser.add_argument('-e', '--epochs',     help='epochs to train',       type=int,   default=10)
    parser.add_argument('-k', '--kfold',      help='kfold cross validation',type=int,   default=10)
    parser.add_argument('-b', '--batch_size', help='batch size per step',   type=int,   default=20)
    parser.add_argument('-f', '--first_decay_epoch', help='epochs for learning rate decay', type=int, default=5)
    parser.add_argument('-i', '--initial_lr', help='initial learning rate', type=float, default=0.005)
    parser.add_argument('-m', '--m_mul',      help='cosine decay param',    type=float, default=0.7)
    parser.add_argument('-a', '--alpha',      help='cosine decay param',    type=float, default=3)
    parser.add_argument('-l', '--lambda',     help='cosine decay param',    type=float, default=0.02, dest='lambda_val')
    parser.add_argument(
        '-c', '--continue', action='store_true', default=False, help='continue from last recorded task', dest='continue_train'
    )
    parser.add_argument('--model_type', help='model type', default='eefm_dual_attention')
    return parser

def load_model(model, model_type=None):
    prev_tasks = Task.get_tasks(project_name='semantic_segmentation', \
                                task_name='cityscapes segmentation', \
                                task_filter= { 
                                    'status':['completed', 'published'],
                                    'order_by':['-last_update']
                                })
    if (len(prev_tasks) == 0):
        print('no valid task from previous experiments')
        return 
    prev_task_id = prev_tasks[0].task_id
    prev_task = Task.get_task(prev_task_id)
    snapshots = [m for m in prev_task.models['output'] if (model_type in m.url or model_type is None)]
    if (len(snapshots) == 0):
        print('no valid model snapshots')
        return 
    print('getting model weights from:', snapshots[-1].url)
    local_weights_path = snapshots[-1].get_local_copy()
    if (local_weights_path is None):
        print('failed to download weights')
        return 
    print('loading weights from:', local_weights_path)
    try:
        model.load_weights(local_weights_path)
    except:
        print('failed loading weights')

def log_pred(image, mask, model, logger, series):
    pred = model.predict(image[tf.newaxis, ...])
    if (isinstance(pred, list)):
        pred = pred[0]
    # print(tf.shape(pred))
    pred_mask = tf.argmax(pred, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis][0]
    
    display_list = [image, mask, pred_mask]
    
    fig = plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    logger.report_matplotlib_figure('Model Prediction', series, fig)

def get_loss(model, l): 
    assert l <= 1, 'auxiliary loss cannot be greater than main loss'
    lossDict = {}
    lossWeights = {}
    for node in model.outputs:
        n = node.name.split('/')[0]
        if n == 'softmax_out':
            lossDict[n] = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, name="main_loss")
            lossWeights[n] = 1.0
        elif 'aux_out' in n:
            try:
                i = int(n.replace('aux_out',''))
                lossDict[n] = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name=f"aux_loss_{i}")
                lossWeights[n] = l**i
            except:
                continue
    if 'softmax_out' not in lossDict.keys():
        n = model.outputs[0].name
        lossDict[n] = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, name="main_loss")
        lossWeights[n] = 1.0

    return lossDict, lossWeights

class LogPlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, image, mask, logger):
        super(LogPlotCallback, self).__init__()
        self.image = image
        self.mask = mask
        self.logger = logger
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch%10==0):
            log_pred(self.image, self.mask, self.model, self.logger, f'epoch: {epoch}')

# define model type options
models = {
    'fpn': get_fpn,
    'enhanced_fpn': get_enhanced_fpn,
    'enhanced_efm': get_enhanced_efm,
    'eefm_attention': get_eefm_attention,
    'eefm_dual_attention': get_eefm_dual_attention,
    'eefm_cross_attention': get_eefm_cross_attention,
    'enhanced_efm_small': get_enhanced_efm_small,
    'efm_v2': get_efm_v2,
    'unet_transformer': get_unet_transformer,
    'm_unet_transformer': get_munet_transformer
}

# start clearml task and get config
if (not DEBUG):
    task = Task.init(project_name='semantic_segmentation', task_name='cityscapes segmentation', output_uri='http://192.168.0.152:8081')
    logger = task.get_logger()

# get dataset
dataset = Dataset.get(dataset_project='semantic_segmentation', dataset_name='cityscapes_fine', dataset_version='1.0.0-c416')
dataset_path = dataset.get_local_copy()

train_images = tf.data.Dataset.list_files(f'{dataset_path}/train/*/*_gtFine_labelIds.png').map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_images = tf.data.Dataset.list_files(f'{dataset_path}/val/*/*_gtFine_labelIds.png').map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()  
    print('configs =',args)
    
    # constants
    MODEL_TYPE = args.model_type
    EPOCHS = args.epochs
    KFOLD = args.kfold
    BATCH_SIZE = args.batch_size
    LAMBDA = args.lambda_val
    VAL_SUBSPLITS = 5
    BUFFER_SIZE = BATCH_SIZE*2
    TRAIN_LENGTH = train_images.cardinality().numpy()
    TEST_LENGTH = test_images.cardinality().numpy()
    SKIP_BATCH = math.ceil(TRAIN_LENGTH/KFOLD/BATCH_SIZE)
    VALIDATION_STEPS = TEST_LENGTH//BATCH_SIZE//VAL_SUBSPLITS
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE - SKIP_BATCH

    # data processing
    train_batches = (
        train_images
        .shuffle(BUFFER_SIZE, seed=0)
        .batch(BATCH_SIZE)
        .repeat()
        .skip(SKIP_BATCH)
        .map(Augment())
        .prefetch(buffer_size=tf.data.AUTOTUNE))
    test_batches = test_images.repeat().batch(BATCH_SIZE)
    sample_image, sample_mask = next(iter(test_images.take(1)))

    # define model
    model = models[MODEL_TYPE](8)

    # load model weights from previous tasks
    if (args.continue_train and not DEBUG):
        load_model(model, MODEL_TYPE)
    analyze(model)
    if (not DEBUG):
        log_pred(sample_image, sample_mask, model, logger, 'start')

    # loss functions
    losses, lossWeights = get_loss(model, LAMBDA)

    # callbacks
    logs = f'{MODEL_TYPE}{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    checkpoint_path = f'{MODEL_TYPE}{datetime.now().strftime("%Y%m%d-%H%M%S")}_ckpt'
    callbacks = []
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                    histogram_freq = 1,
                                                    profile_batch = '200,220'))
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    verbose=1))
    if (not DEBUG):
        callbacks.append(LogPlotCallback(sample_image, sample_mask, logger))
    # learning rate schedule
    lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecayRestarts(
        args.initial_lr,
        args.first_decay_epoch*STEPS_PER_EPOCH,
        m_mul=args.m_mul,
        alpha=10**-args.alpha)


    # model training
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_decayed_fn),
                loss=losses, loss_weights=lossWeights,
                metrics=['sparse_categorical_accuracy','mean_squared_error',SparseMeanIoU(num_classes=8)])

    model_history = model.fit(train_batches, epochs=EPOCHS,
                            steps_per_epoch=STEPS_PER_EPOCH,
                            validation_steps=VALIDATION_STEPS,
                            validation_data=test_batches,
                            callbacks=callbacks)
    
    model.load_weights(checkpoint_path)
    model.save(MODEL_TYPE)
    if (not DEBUG):
        log_pred(sample_image, sample_mask, model, logger, 'end')
        task.close()