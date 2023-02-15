import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from clearml import Task, Dataset
from utils import *

# device capabilities
physical_devices = tf.config.list_physical_devices('GPU')
print(tf.__version__, physical_devices)
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


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
    snapshots = [m for m in prev_task.models['output'] if (model_type in m or model_type is None)]
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
        model.load_weights('enhanced_efm20230215-160814')
        print('fail loading weights')

def log_pred(image, mask, model, logger, series):
    pred = model.predict(sample_image[tf.newaxis, ...])
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
    
class LogPlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, image, mask, logger):
        super(LogPlotCallback, self).__init__()
        self.image = image
        self.mask = mask
        self.logger = logger
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch%10==0):
            log_pred(self.image, self.mask, self.model, self.logger, f'epoch: {epoch}')

# start clearml task and get config
task = Task.init(project_name='semantic_segmentation', task_name='cityscapes segmentation', output_uri='http://192.168.0.152:8081')
logger = task.get_logger()
configs = {'epochs': 200, 'batch_size': 20, 'base_lr': 0.001, 'first_decay_epoch': 5,'model_type': 'enhanced_efm', 'continue': True}
configs = task.connect(configs) 
print('configs =', configs) 

# get dataset
dataset = Dataset.get(dataset_project='semantic_segmentation', dataset_name='cityscapes_fine', dataset_version='1.0.0-c416')
dataset_path = dataset.get_local_copy()

train_images = tf.data.Dataset.list_files(f'{dataset_path}/train/*/*_gtFine_labelIds.png').map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_images = tf.data.Dataset.list_files(f'{dataset_path}/val/*/*_gtFine_labelIds.png').map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

# constants
EPOCHS = configs['epochs']
BATCH_SIZE = configs['batch_size']
VAL_SUBSPLITS = 5
BUFFER_SIZE = BATCH_SIZE*2
TRAIN_LENGTH = train_images.cardinality().numpy()
TEST_LENGTH = test_images.cardinality().numpy()
VALIDATION_STEPS = TEST_LENGTH//BATCH_SIZE//VAL_SUBSPLITS
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

# data processing
train_batches = (
    train_images
    .shuffle(BUFFER_SIZE, seed=0)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE))
test_batches = test_images.batch(BATCH_SIZE)
sample_image, sample_mask = next(iter(test_images.take(1)))

# define model
models = {
    'fpn': get_fpn,
    'enhanced_fpn': get_enhanced_fpn,
    'enhanced_efm': get_enhanced_efm,
    'enhanced_efm_small': get_enhanced_efm_small
}
model = models[configs['model_type']](8)

# load model weights from previous tasks
if (configs['continue']):
    load_model(model, configs['model_type'])
analyze(model)
log_pred(sample_image, sample_mask, model, logger, 'start')

# callbacks
logs = f'{configs["model_type"]}{datetime.now().strftime("%Y%m%d-%H%M%S")}'
checkpoint_path = f'{configs["model_type"]}{datetime.now().strftime("%Y%m%d-%H%M%S")}_ckpt'

tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                histogram_freq = 1,
                                                profile_batch = '200,220')
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 verbose=1)
lp_callback = LogPlotCallback(sample_image, sample_mask, logger)
# learning rate schedule
lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecayRestarts(
    configs['base_lr'],
    configs['first_decay_epoch']*STEPS_PER_EPOCH,
    m_mul=0.9,
    alpha=1e-3)
# IoU metric for sparse category, IoU = TP/(TP+FP+FN)
class SparseMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self,
               y_true=None,
               y_pred=None,
               num_classes=None,
               name='sparse_mean_iou',
               dtype=None):
        super(SparseMeanIoU, self).__init__(num_classes = num_classes,name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)
    
# model training
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_decayed_fn),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['sparse_categorical_accuracy','mean_squared_error',SparseMeanIoU(num_classes=8)])

model_history = model.fit(train_batches, epochs=EPOCHS,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        validation_steps=VALIDATION_STEPS,
                        validation_data=test_batches,
                        callbacks=[tboard_callback, cp_callback, lp_callback])

log_pred(sample_image, sample_mask, model, logger, 'end')
model.save(configs["model_type"])
task.close()